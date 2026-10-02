from typing import Optional
import torch
from einops import rearrange
from torch import Tensor
from ..utils import exists


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def to_batch(
    batch_size: int,
    device: torch.device,
    x: Optional[float | Tensor] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    if exists(x):
        if isinstance(x, Tensor):
            if x.numel() == 1:
                x = x.to(device) if x.device != device else x
                xs = x.expand(batch_size)
            else:
                raise ValueError("x must be a scalar tensor or float")
        else:
            xs = torch.full(size=(batch_size,), fill_value=x, device=device)
    assert exists(xs)
    return xs
