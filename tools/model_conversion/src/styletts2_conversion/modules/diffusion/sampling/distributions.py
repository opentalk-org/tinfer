import torch
from torch import Tensor
from math import pi


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


class UniformDistribution(Distribution):
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        return torch.rand(num_samples, device=device)


class VKDistribution(Distribution):
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = float("inf"),
        sigma_data: float = 1.0,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        sigma_data = self.sigma_data
        min_cdf = torch.atan(torch.tensor(self.min_value / sigma_data)) * 2 / pi
        max_cdf = torch.atan(torch.tensor(self.max_value / sigma_data)) * 2 / pi
        u = (max_cdf - min_cdf) * torch.randn((num_samples,), device=device) + min_cdf
        return torch.tan(u * pi / 2) * sigma_data
