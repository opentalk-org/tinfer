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
