from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..utils import exists, default
from .process import Diffusion, KDiffusion, VDiffusion, VKDiffusion
from .schedules import Schedule
from .samplers import Sampler


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = None,
        clamp: bool = True,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

        # Check sampler is compatible with diffusion type
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f"{sampler_class} incompatible with {diffusion_class}"
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0) if self.clamp else x
        return x


class DiffusionInpainter(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        num_steps: int,
        num_resamples: int,
        sampler: Sampler,
        sigma_schedule: Schedule,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        self.num_resamples = num_resamples
        self.inpaint_fn = sampler.inpaint
        self.sigma_schedule = sigma_schedule

    @torch.no_grad()
    def forward(self, inpaint: Tensor, inpaint_mask: Tensor) -> Tensor:
        x = self.inpaint_fn(
            source=inpaint,
            mask=inpaint_mask,
            fn=self.denoise_fn,
            sigmas=self.sigma_schedule(self.num_steps, inpaint.device),
            num_steps=self.num_steps,
            num_resamples=self.num_resamples,
        )
        return x


def sequential_mask(like: Tensor, start: int) -> Tensor:
    length, device = like.shape[2], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[:, :, start:] = torch.zeros((length - start,), device=device)
    return mask


class SpanBySpanComposer(nn.Module):
    def __init__(
        self,
        inpainter: DiffusionInpainter,
        *,
        num_spans: int,
    ):
        super().__init__()
        self.inpainter = inpainter
        self.num_spans = num_spans

    def forward(self, start: Tensor, keep_start: bool = False) -> Tensor:
        half_length = start.shape[2] // 2

        spans = list(start.chunk(chunks=2, dim=-1)) if keep_start else []
        # Inpaint second half from first half
        inpaint = torch.zeros_like(start)
        inpaint[:, :, :half_length] = start[:, :, half_length:]
        inpaint_mask = sequential_mask(like=start, start=half_length)

        for i in range(self.num_spans):
            # Inpaint second half
            span = self.inpainter(inpaint=inpaint, inpaint_mask=inpaint_mask)
            # Replace first half with generated second half
            second_half = span[:, :, half_length:]
            inpaint[:, :, :half_length] = second_half
            # Save generated span
            spans.append(second_half)

        return torch.cat(spans, dim=2)


class XDiffusion(nn.Module):
    def __init__(self, type: str, net: nn.Module, **kwargs):
        super().__init__()

        diffusion_classes = [VDiffusion, KDiffusion, VKDiffusion]
        aliases = [t.alias for t in diffusion_classes]  # type: ignore
        message = f"type='{type}' must be one of {*aliases,}"
        assert type in aliases, message
        self.net = net

        for XDiffusion in diffusion_classes:
            if XDiffusion.alias == type:  # type: ignore
                self.diffusion = XDiffusion(net=net, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(*args, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        clamp: bool,
        **kwargs,
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
            clamp=clamp,
        )
        return diffusion_sampler(noise, **kwargs)
