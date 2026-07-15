from __future__ import annotations

import math
import os
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm

from .tensorrt_runtime import (
    DynamicDecoderTRTEngineSpec,
    DynamicDiffusionTRTEngineSpec,
    ShapeProfile,
    decoder_dynamic_profile_shapes,
    diffusion_dynamic_profile_shapes,
)


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


def export_decoder_dynamic_onnx(
    decoder: nn.Module,
    output_dir: str | Path,
    *,
    opt_batch_size: int,
    opt_asr_frames: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    opset_version: int = 20,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = DynamicDecoderTRTEngineSpec()
    output_path = output_dir / spec.onnx_name

    decoder = decoder.eval().to(device)
    remove_decoder_weight_norm(decoder)
    if dtype == torch.float16:
        decoder = decoder.half()

    f0_frames = opt_asr_frames * 2
    har_frames = opt_asr_frames * 120 + 1
    asr = torch.randn(opt_batch_size, 512, opt_asr_frames, device=device, dtype=dtype)
    f0 = torch.rand(opt_batch_size, f0_frames, device=device, dtype=dtype) * 120.0 + 80.0
    noise = torch.randn(opt_batch_size, f0_frames, device=device, dtype=dtype)
    style = torch.randn(opt_batch_size, 128, device=device, dtype=dtype)
    har = torch.randn(opt_batch_size, 22, har_frames, device=device, dtype=dtype)

    wrapper = DecoderTRTExportModule(decoder).eval()
    old_export_flag = os.environ.get("TINFER_TRT_EXPORT")
    os.environ["TINFER_TRT_EXPORT"] = "1"
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (asr, f0, noise, style, har),
                str(output_path),
                input_names=["asr", "f0", "noise", "style", "har"],
                output_names=["audio"],
                opset_version=opset_version,
                dynamo=False,
                do_constant_folding=True,
                dynamic_axes={
                    "asr": {0: "batch", 2: "asr_frames"},
                    "f0": {0: "batch", 1: "f0_frames"},
                    "noise": {0: "batch", 1: "f0_frames"},
                    "style": {0: "batch"},
                    "har": {0: "batch", 2: "har_frames"},
                    "audio": {0: "batch", 2: "audio_samples"},
                },
            )
    finally:
        if old_export_flag is None:
            os.environ.pop("TINFER_TRT_EXPORT", None)
        else:
            os.environ["TINFER_TRT_EXPORT"] = old_export_flag

    return output_path


def export_diffusion_dynamic_onnx(
    diffusion: nn.Module,
    output_dir: str | Path,
    *,
    opt_batch_size: int,
    opt_embedding_tokens: int,
    num_steps: int = 10,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    opset_version: int = 20,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = DynamicDiffusionTRTEngineSpec(num_steps=num_steps)
    output_path = output_dir / spec.onnx_name

    diffusion = diffusion.eval().to(device)
    if dtype == torch.float16:
        diffusion = diffusion.half()

    noise = torch.randn(opt_batch_size, 1, 256, device=device, dtype=dtype)
    step_noise = torch.randn(opt_batch_size, num_steps - 1, 1, 256, device=device, dtype=dtype)
    embedding = torch.randn(opt_batch_size, opt_embedding_tokens, 768, device=device, dtype=dtype)
    features = torch.randn(opt_batch_size, 256, device=device, dtype=dtype)

    wrapper = DiffusionTRTExportModule(diffusion, num_steps=num_steps).eval()
    old_export_flag = os.environ.get("TINFER_TRT_EXPORT")
    os.environ["TINFER_TRT_EXPORT"] = "1"
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (noise, step_noise, embedding, features),
                str(output_path),
                input_names=["noise", "step_noise", "embedding", "features"],
                output_names=["style"],
                opset_version=opset_version,
                dynamo=False,
                do_constant_folding=True,
                dynamic_axes={
                    "noise": {0: "batch"},
                    "step_noise": {0: "batch"},
                    "embedding": {0: "batch", 1: "embedding_tokens"},
                    "features": {0: "batch"},
                    "style": {0: "batch"},
                },
            )
    finally:
        if old_export_flag is None:
            os.environ.pop("TINFER_TRT_EXPORT", None)
        else:
            os.environ["TINFER_TRT_EXPORT"] = old_export_flag

    return output_path


def _build_engine_from_onnx(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    workspace_bytes: int,
    profile_shapes: ShapeProfile | None = None,
) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = 5
    if profile_shapes is not None:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in profile_shapes.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return engine_path


def build_decoder_dynamic_engine_from_onnx(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 16,
    min_asr_frames: int = 128,
    opt_asr_frames: int = 256,
    max_asr_frames: int = 512,
    workspace_bytes: int = 4 << 30,
) -> Path:
    profile_shapes = decoder_dynamic_profile_shapes(
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        min_asr_frames=min_asr_frames,
        opt_asr_frames=opt_asr_frames,
        max_asr_frames=max_asr_frames,
    )
    return _build_engine_from_onnx(
        onnx_path,
        engine_path,
        workspace_bytes=workspace_bytes,
        profile_shapes=profile_shapes,
    )


def build_diffusion_dynamic_engine_from_onnx(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 16,
    min_tokens: int = 16,
    opt_tokens: int = 128,
    max_tokens: int = 256,
    num_steps: int = 10,
    workspace_bytes: int = 4 << 30,
) -> Path:
    profile_shapes = diffusion_dynamic_profile_shapes(
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        min_tokens=min_tokens,
        opt_tokens=opt_tokens,
        max_tokens=max_tokens,
        num_steps=num_steps,
    )
    return _build_engine_from_onnx(
        onnx_path,
        engine_path,
        workspace_bytes=workspace_bytes,
        profile_shapes=profile_shapes,
    )
