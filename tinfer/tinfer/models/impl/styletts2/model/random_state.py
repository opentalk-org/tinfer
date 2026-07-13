import torch


def request_generator(
    seed: int | None,
    device: torch.device,
) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
