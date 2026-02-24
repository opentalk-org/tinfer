from __future__ import annotations
from dataclasses import dataclass


@dataclass
class StyleTTS2Params:
    use_diffusion: bool = True
    phonemized: bool = False
    embedding_scale: float = 1.0
    diffusion_steps: int = 10
    style_interpolation_factor: float = 0.0
    alpha: float = 0.3
    beta: float = 0.7
    speed: float = 1.0


