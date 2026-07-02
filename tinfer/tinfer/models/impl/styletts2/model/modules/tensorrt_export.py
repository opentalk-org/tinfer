"""Model-owned TensorRT export adapters for StyleTTS2.

Only the graph rewrites and wrapper modules that ONNX export needs live here;
what to export, with which shapes, is declared in trt_bundle.py, and the
export/build mechanics are trtc's.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm


def _irfft20_basis(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    n = torch.arange(20, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(11, device=device, dtype=dtype).unsqueeze(0)
    angle = 2.0 * math.pi * n * k / 20.0
    scale = torch.full((1, 11), 2.0 / 20.0, device=device, dtype=dtype)
    scale[:, 0] = 1.0 / 20.0
    scale[:, 10] = 1.0 / 20.0
    cos_basis = torch.cos(angle) * scale
    sin_basis = -torch.sin(angle) * scale
    return cos_basis, sin_basis


def onnx_istft20_inverse(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    window: torch.Tensor,
    *,
    hop_length: int = 5,
) -> torch.Tensor:
    """torch.istft equivalent for n_fft=20 without complex ops, which ONNX and
    TensorRT cannot represent: irFFT as real cos/sin matmuls + overlap-add."""
    if magnitude.shape[1] != 11 or phase.shape[1] != 11:
        raise ValueError("onnx_istft20_inverse expects 11 frequency bins for n_fft=20")

    dtype = magnitude.dtype
    device = magnitude.device
    window = window.to(device=device, dtype=dtype)
    cos_basis, sin_basis = _irfft20_basis(device, dtype)

    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    frames = torch.einsum("bft,nf->bnt", real, cos_basis) + torch.einsum("bft,nf->bnt", imag, sin_basis)
    frames = frames * window.view(1, 20, 1)

    ola_weight = torch.eye(20, device=device, dtype=dtype).view(20, 1, 20)
    audio = F.conv_transpose1d(frames, ola_weight, stride=hop_length)

    norm_input = torch.ones(
        (magnitude.shape[0], 1, magnitude.shape[2]),
        device=device,
        dtype=dtype,
    )
    norm_weight = (window.square()).view(1, 1, 20)
    norm = F.conv_transpose1d(norm_input, norm_weight, stride=hop_length)
    audio = audio / torch.clamp(norm, min=1e-8)

    center = 10
    return audio[:, :, center:-center]


class DecoderTRTExportModule(nn.Module):
    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        asr: torch.Tensor,
        f0: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
        har: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder.forward_with_har(asr, f0, noise, style, har)


class DiffusionTRTExportModule(nn.Module):
    def __init__(
        self,
        diffusion: nn.Module,
        *,
        num_steps: int,
        sigma_min: float = 0.0001,
        sigma_max: float = 3.0,
        schedule_rho: float = 9.0,
        sampler_rho: float = 1.0,
        embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("Diffusion TensorRT export requires num_steps >= 2")
        self.diffusion = diffusion
        self.num_steps = int(num_steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.schedule_rho = float(schedule_rho)
        self.sampler_rho = float(sampler_rho)
        self.embedding_scale = float(embedding_scale)

    def _sigmas(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rho_inv = 1.0 / self.schedule_rho
        steps = torch.arange(self.num_steps, device=device, dtype=dtype)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (self.num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.schedule_rho
        return F.pad(sigmas, pad=(0, 1), value=0.0)

    def _denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = sigma.expand(x.shape[0])
        return self.diffusion.denoise_fn(
            x,
            sigmas=sigmas,
            embedding=embedding,
            features=features,
            embedding_scale=self.embedding_scale,
        )

    def _adpm2_step(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        step_noise: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        sampler_rho = torch.tensor(self.sampler_rho, device=x.device, dtype=x.dtype)
        sigma_up = torch.sqrt(sigma_next.square() * (sigma.square() - sigma_next.square()) / sigma.square())
        sigma_down = torch.sqrt(sigma_next.square() - sigma_up.square())
        sigma_mid = ((sigma ** (1 / sampler_rho) + sigma_down ** (1 / sampler_rho)) / 2) ** sampler_rho

        d = (x - self._denoise(x, sigma, embedding, features)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - self._denoise(x_mid, sigma_mid, embedding, features)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        return x + step_noise * sigma_up

    def forward(
        self,
        noise: torch.Tensor,
        step_noise: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = self._sigmas(noise.device, noise.dtype)
        x = sigmas[0] * noise
        for index in range(self.num_steps - 1):
            x = self._adpm2_step(
                x,
                sigmas[index],
                sigmas[index + 1],
                step_noise[:, index],
                embedding,
                features,
            )
        return x


def remove_decoder_weight_norm(decoder: nn.Module) -> None:
    for module in decoder.modules():
        try:
            remove_weight_norm(module)
        except ValueError:
            pass
