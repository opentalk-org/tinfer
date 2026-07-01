from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm

from .tensorrt_runtime import DecoderTRTEngineSpec


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


def remove_decoder_weight_norm(decoder: nn.Module) -> None:
    for module in decoder.modules():
        try:
            remove_weight_norm(module)
        except ValueError:
            pass


def export_decoder_onnx(
    decoder: nn.Module,
    output_dir: str | Path,
    *,
    batch_size: int,
    asr_frames: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    opset_version: int = 20,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = DecoderTRTEngineSpec(batch_size=batch_size, asr_frames=asr_frames)
    output_path = output_dir / spec.onnx_name

    decoder = decoder.eval().to(device)
    remove_decoder_weight_norm(decoder)
    if dtype == torch.float16:
        decoder = decoder.half()

    f0_frames = asr_frames * 2
    har_frames = f0_frames * 60 // 1 + 1
    asr = torch.randn(batch_size, 512, asr_frames, device=device, dtype=dtype)
    f0 = torch.rand(batch_size, f0_frames, device=device, dtype=dtype) * 120.0 + 80.0
    noise = torch.randn(batch_size, f0_frames, device=device, dtype=dtype)
    style = torch.randn(batch_size, 128, device=device, dtype=dtype)
    har = torch.randn(batch_size, 22, har_frames, device=device, dtype=dtype)

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
            )
    finally:
        if old_export_flag is None:
            os.environ.pop("TINFER_TRT_EXPORT", None)
        else:
            os.environ["TINFER_TRT_EXPORT"] = old_export_flag

    return output_path


def build_decoder_engine_from_onnx(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    workspace_bytes: int = 4 << 30,
) -> Path:
    import tensorrt as trt

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
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return engine_path
