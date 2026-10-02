from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def stft20(source: torch.Tensor, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 2:
        raise ValueError("stft20 expects [batch, samples]")
    dtype = source.dtype
    device = source.device
    samples = F.pad(source.unsqueeze(1), (10, 10), mode="reflect")
    time = torch.arange(20, device=device, dtype=dtype).view(1, 20)
    frequency = torch.arange(11, device=device, dtype=dtype).view(11, 1)
    angle = 2.0 * math.pi * frequency * time / 20.0
    window = window.to(device=device, dtype=dtype).view(1, 20)
    filters = torch.cat((torch.cos(angle) * window, -torch.sin(angle) * window))
    transformed = F.conv1d(samples, filters.unsqueeze(1), stride=5)
    real = transformed[:, :11]
    imaginary = transformed[:, 11:]
    magnitude = torch.sqrt(real.square() + imaginary.square())
    return magnitude, torch.atan2(imaginary, real)


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
