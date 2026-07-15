from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm

from tinfer.models.impl.styletts2.model.modules import istftnet
from tinfer.models.impl.styletts2.model.modules.blocks import AdaLayerNorm
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import DecoderBackbone
from tools.styletts2_model_scripts.fourier import stft20


def strip_weight_norm(module: nn.Module) -> None:
    for child in module.modules():
        try:
            remove_weight_norm(child)
        except ValueError:
            pass


def guided_output(conditional: torch.Tensor, unconditional: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.reshape(-1, 1, 1).to(conditional.dtype)
    return unconditional + (conditional - unconditional) * scale


def blend_style(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    use_diffusion: torch.Tensor,
    previous: torch.Tensor,
    has_previous: torch.Tensor,
    interpolation: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(candidate).all(dim=1) & (torch.linalg.vector_norm(candidate.float(), dim=1) <= 10.0)
    candidate = torch.where((use_diffusion & valid)[:, None], candidate, reference)
    mixed = interpolation[:, None] * previous + (1.0 - interpolation[:, None]) * candidate
    candidate = torch.where((use_diffusion & has_previous)[:, None], mixed, candidate)
    ref = alpha * candidate[:, :128] + (1.0 - alpha) * reference[:, :128]
    style = beta * candidate[:, 128:] + (1.0 - beta) * reference[:, 128:]
    return style, ref


def native_lstm(lstm: nn.LSTM, value: torch.Tensor) -> torch.Tensor:
    directions = 2 if lstm.bidirectional else 1
    state_shape = (directions * lstm.num_layers, value.shape[0], lstm.hidden_size)
    hidden = value.new_zeros(state_shape)
    cell = value.new_zeros(state_shape)
    return lstm(value, (hidden, cell))[0]


def encode_text(encoder: nn.Module, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = encoder.embedding(tokens).transpose(1, 2)
    expanded_mask = mask.unsqueeze(1)
    value = value.masked_fill(expanded_mask, 0.0)
    for convolution in encoder.cnn:
        value = convolution(value).masked_fill(expanded_mask, 0.0)
    value = native_lstm(encoder.lstm, value.transpose(1, 2)).transpose(1, 2)
    return value.masked_fill(expanded_mask, 0.0)


def encode_duration(encoder: nn.Module, value: torch.Tensor, style: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = value.permute(2, 0, 1)
    expanded_style = style.expand(value.shape[0], value.shape[1], -1)
    value = torch.cat((value, expanded_style), dim=-1)
    value = value.masked_fill(mask.transpose(0, 1).unsqueeze(-1), 0.0)
    value = value.transpose(0, 1).transpose(1, 2)
    for block in encoder.lstms:
        if isinstance(block, AdaLayerNorm):
            value = block(value.transpose(1, 2), style).transpose(1, 2)
            value = torch.cat((value, expanded_style.permute(1, 2, 0)), dim=1)
            value = value.masked_fill(mask.unsqueeze(1), 0.0)
        else:
            value = native_lstm(block, value.transpose(1, 2)).transpose(1, 2)
    return value.transpose(1, 2)


class EngineA(nn.Module):
    def __init__(self, model: nn.Module, max_steps: int) -> None:
        super().__init__()
        self.text_encoder = model.text_encoder
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor = model.predictor
        self.diffusion = model.diffusion.diffusion
        self.max_steps = max_steps

    def denoise(
        self,
        value: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = sigma.expand(value.shape[0])
        skip, output, input_scale, noise = self.diffusion.get_scale_weights(sigmas)
        network = self.diffusion.net
        network_input = (input_scale * value).to(embedding.dtype)
        fixed = network.fixed_embedding(embedding)
        conditional = network.run(network_input, noise.to(embedding.dtype), embedding=embedding, features=features)
        unconditional = network.run(network_input, noise.to(embedding.dtype), embedding=fixed, features=features)
        prediction = guided_output(conditional, unconditional, scale)
        return skip * value + output * prediction.float()

    def diffusion_step(
        self,
        value: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        noise: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        sigma_up = torch.sqrt(next_sigma.square() * (sigma.square() - next_sigma.square()) / sigma.square())
        sigma_down = torch.sqrt(next_sigma.square() - sigma_up.square())
        midpoint = (sigma + sigma_down) / 2.0
        derivative = (value - self.denoise(value, sigma, embedding, features, scale)) / sigma
        middle = value + derivative * (midpoint - sigma)
        middle_derivative = (middle - self.denoise(middle, midpoint, embedding, features, scale)) / midpoint
        return value + middle_derivative * (sigma_down - sigma) + noise * sigma_up

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        ref_s: torch.Tensor,
        noise: torch.Tensor,
        step_noise: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        scale: torch.Tensor,
        sigmas: torch.Tensor,
        use_diffusion: torch.Tensor,
        previous_s: torch.Tensor,
        has_previous: torch.Tensor,
        style_interpolation: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        text = encode_text(self.text_encoder, tokens, mask)
        bert = self.bert(tokens, attention_mask=(~mask).int()).to(text.dtype)
        duration_encoding = self.bert_encoder(bert).transpose(1, 2)
        value = sigmas[0].float() * noise.float()
        splits = []
        for index in range(self.max_steps - 1):
            value = self.diffusion_step(
                value, sigmas[index].float(), sigmas[index + 1].float(), step_noise[:, index].float(), bert, ref_s, scale
            )
            splits.append(value.to(text.dtype))
        style, ref = blend_style(
            value.squeeze(1).to(text.dtype), ref_s, use_diffusion, previous_s, has_previous, style_interpolation, alpha, beta
        )
        encoded = encode_duration(self.predictor.text_encoder, duration_encoding, style, mask)
        duration = torch.sigmoid(self.predictor.duration_proj(native_lstm(self.predictor.lstm, encoded))).sum(dim=-1)
        return (duration, encoded, text, style, ref, bert, duration_encoding, *splits)


class EngineBC(nn.Module):
    def __init__(self, predictor: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.decoder = decoder

    def _prosody(self, encoding: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared, _ = self.predictor.shared(encoding.transpose(1, 2))
        f0 = shared.transpose(1, 2)
        noise = shared.transpose(1, 2)
        for block in self.predictor.F0:
            f0 = block(f0, style)
        for block in self.predictor.N:
            noise = block(noise, style)
        return self.predictor.F0_proj(f0).squeeze(1), self.predictor.N_proj(noise).squeeze(1)

    def _harmonic(
        self,
        f0: torch.Tensor,
        phase: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_f0 = f0.float().unsqueeze(2)
        harmonics = torch.arange(1, 10, device=f0.device, dtype=torch.float32).view(1, 1, 9)
        increments = phase_f0 * harmonics * (2.0 * torch.pi / 24000.0)
        phases = phase.float().unsqueeze(1) + torch.cumsum(increments, dim=1)
        sample_phases = F.interpolate(
            phases.transpose(1, 2), size=105600, mode="linear", align_corners=False
        ).transpose(1, 2) * 300.0
        voiced = F.interpolate(f0.unsqueeze(1), size=105600, mode="nearest").transpose(1, 2) > 10.0
        waves = torch.sin(sample_phases).to(f0.dtype) * 0.1
        voiced_noise = source_noise * 0.003
        unvoiced_noise = source_noise * (0.1 / 3.0)
        source = torch.where(voiced, waves + voiced_noise, unvoiced_noise)
        source = self.decoder.generator.m_source.l_tanh(
            self.decoder.generator.m_source.l_linear(source)
        ).squeeze(2)
        magnitude, source_phase = stft20(source, self.decoder.generator.stft.window)
        harmonic = torch.cat((magnitude, source_phase), dim=1)
        next_phase = phases[:, 255, :]
        next_phase = next_phase - torch.floor(next_phase / (2.0 * torch.pi)) * (2.0 * torch.pi)
        return harmonic, next_phase

    def forward(
        self,
        encoding: torch.Tensor,
        asr: torch.Tensor,
        style: torch.Tensor,
        reference: torch.Tensor,
        phase: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f0, noise = self._prosody(encoding, style)
        harmonic, next_phase = self._harmonic(f0, phase, source_noise)
        generator = self.decoder.generator
        value, _ = DecoderBackbone.forward(self.decoder, asr, f0, noise, reference)
        for index in range(generator.num_upsamples):
            value = F.leaky_relu(value, istftnet.LRELU_SLOPE)
            source = generator.noise_res[index](generator.noise_convs[index](harmonic), reference)
            value = generator.ups[index](value)
            if index == generator.num_upsamples - 1:
                value = generator.reflection_pad(value)
            value = generator._process_resblocks(value + source, reference, index)
        audio = generator._generate_output(value)
        return audio[:, :, 19200:96000], next_phase
