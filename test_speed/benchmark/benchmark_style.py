from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import torch

from test_speed.benchmark.benchmark_data import RequestMetric


class StyleVectorModel(Protocol):
    def get_voice(self, voice_id: str) -> torch.Tensor: ...


class StyleNormPart(Enum):
    FULL = "full_norm"
    FIRST_HALF = "first_half_norm"
    SECOND_HALF = "second_half_norm"


@dataclass(frozen=True)
class StyleEmbeddingNorm:
    voice_id: str
    full_norm: float
    first_half_norm: float
    second_half_norm: float


def measure_style_norms(
    model: StyleVectorModel,
    voice_ids: list[str],
) -> list[StyleEmbeddingNorm]:
    norms = []
    for voice_id in voice_ids:
        vector = model.get_voice(voice_id).detach().flatten().cpu()
        assert vector.numel() == 256, f"{voice_id} style vector is not 256 values"
        norms.append(
            StyleEmbeddingNorm(
                voice_id,
                float(torch.linalg.vector_norm(vector)),
                float(torch.linalg.vector_norm(vector[:128])),
                float(torch.linalg.vector_norm(vector[128:])),
            )
        )
    return norms


def style_norm_coordinates(
    metrics: list[RequestMetric],
    norms: list[StyleEmbeddingNorm],
    part: StyleNormPart,
) -> tuple[list[int], list[float]]:
    values = {
        item.voice_id: float(getattr(item, part.value))
        for item in norms
    }
    return (
        [metric.input_phoneme_tokens for metric in metrics],
        [values[metric.voice_id] for metric in metrics],
    )


def _plot_style_norm(
    metrics: list[RequestMetric],
    norms: list[StyleEmbeddingNorm],
    part: StyleNormPart,
    title: str,
    path: Path,
) -> Path:
    x_values, y_values = style_norm_coordinates(metrics, norms, part)
    assert x_values, f"Cannot plot empty style norms: {title}"
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(x_values, y_values, alpha=0.65)
    axis.set(
        title=title,
        xlabel="Input phoneme tokens",
        ylabel="Style embedding L2 norm",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_style_norm_plots(
    metrics: list[RequestMetric],
    norms: list[StyleEmbeddingNorm],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_style_norm(
            metrics,
            norms,
            StyleNormPart.FULL,
            "Input length vs full style embedding norm",
            output_dir / "style_embedding_full_norm_vs_input_tokens.png",
        ),
        _plot_style_norm(
            metrics,
            norms,
            StyleNormPart.FIRST_HALF,
            "Input length vs first 128 style values norm",
            output_dir / "style_embedding_first_128_norm_vs_input_tokens.png",
        ),
        _plot_style_norm(
            metrics,
            norms,
            StyleNormPart.SECOND_HALF,
            "Input length vs second 128 style values norm",
            output_dir / "style_embedding_second_128_norm_vs_input_tokens.png",
        ),
    ]
