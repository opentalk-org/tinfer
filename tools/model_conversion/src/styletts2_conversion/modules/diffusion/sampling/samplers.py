from typing import Callable, List, Tuple, Type
import torch
import torch.nn as nn
from torch import Tensor
from .process import Diffusion, KDiffusion, VDiffusion, VKDiffusion
from math import pi


class Sampler(nn.Module):

    diffusion_types: List[Type[Diffusion]] = []

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")


class VSampler(Sampler):

    diffusion_types = [VDiffusion]

    def get_alpha_beta(self, sigma: Tensor) -> Tuple[Tensor, Tensor]:
        sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma)
        angle = sigma * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        alpha, beta = self.get_alpha_beta(sigmas[0])

        for i in range(num_steps - 1):
            is_last = i == num_steps - 1

            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha

            if not is_last:
                alpha, beta = self.get_alpha_beta(sigmas[i + 1])
                x = x_pred * alpha + x_eps * beta

        return x_pred


class KarrasSampler(Sampler):
    """https://arxiv.org/abs/2206.00364 algorithm 1"""

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(
        self, x: Tensor, fn: Callable, sigma: Tensor, sigma_next: Tensor, gamma: Tensor
    ) -> Tensor:
        sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma, device=x.device, dtype=x.dtype)
        sigma_next = sigma_next if isinstance(sigma_next, Tensor) else torch.tensor(sigma_next, device=x.device, dtype=x.dtype)
        gamma = gamma if isinstance(gamma, Tensor) else torch.tensor(gamma, device=x.device, dtype=x.dtype)
        sigma_hat = sigma + gamma * sigma
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + torch.sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if (sigma_next != 0).any() if isinstance(sigma_next, Tensor) else sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        sqrt2_minus_1 = torch.sqrt(torch.tensor(2.0, device=sigmas.device, dtype=sigmas.dtype)) - 1.0
        gamma_val = torch.minimum(torch.tensor(self.s_churn / num_steps, device=sigmas.device, dtype=sigmas.dtype), sqrt2_minus_1)
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            gamma_val,
            torch.tensor(0.0, device=sigmas.device, dtype=sigmas.dtype),
        )
        for i in range(num_steps - 1):
            x = self.step(
                x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i]
            )

        return x


class AEulerSampler(Sampler):

    diffusion_types = [KDiffusion, VKDiffusion]

    def get_sigmas(self, sigma: Tensor, sigma_next: Tensor) -> Tuple[Tensor, Tensor]:
        sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma, device=sigma_next.device if isinstance(sigma_next, Tensor) else None)
        sigma_next = sigma_next if isinstance(sigma_next, Tensor) else torch.tensor(sigma_next, device=sigma.device)
        sigma_up = torch.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: Tensor, sigma_next: Tensor) -> Tensor:
        sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma, device=x.device, dtype=x.dtype)
        sigma_next = sigma_next if isinstance(sigma_next, Tensor) else torch.tensor(sigma_next, device=x.device, dtype=x.dtype)
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        d = (x - fn(x, sigma=sigma)) / sigma
        x_next = x + d * (sigma_down - sigma)
        noise = torch.randn_like(x)
        x_next = x_next + noise * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
        return x


class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, rho: float = 1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma: Tensor, sigma_next: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma, device=sigma_next.device if isinstance(sigma_next, Tensor) else None)
        sigma_next = sigma_next if isinstance(sigma_next, Tensor) else torch.tensor(sigma_next, device=sigma.device)
        r = torch.tensor(self.rho, device=sigma.device)
        sigma_up = torch.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: Tensor, sigma_next: Tensor) -> Tensor:
        sigma = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma, device=x.device, dtype=x.dtype)
        sigma_next = sigma_next if isinstance(sigma_next, Tensor) else torch.tensor(sigma_next, device=x.device, dtype=x.dtype)
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        d = (x - fn(x, sigma=sigma)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        noise = torch.randn_like(x)
        x_next = x + noise * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        for i in range(num_steps - 1):

            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        noise_init = torch.randn_like(source).clone()
        x = (sigmas[0] * noise_init).clone()

        for i in range(num_steps - 1):
            noise_source = torch.randn_like(source).clone()
            source_noisy = (source + sigmas[i] * noise_source).clone()
            for r in range(num_resamples):
                x = (source_noisy * mask + x * ~mask).clone()
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1]).clone()
                if r < num_resamples - 1:
                    sigma = torch.sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    noise = torch.randn_like(x).clone()
                    x = (x + sigma * noise).clone()

        return (source * mask + x * ~mask).clone()
