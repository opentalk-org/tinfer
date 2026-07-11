from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from test_speed.duration_padding.analysis import DurationTrace, compare_real_tokens


@dataclass(frozen=True)
class DifferenceRow:
    padding_tokens: int
    stage: str
    max_absolute_difference: float
    mean_absolute_difference: float


def build_rows(
    traces: dict[int, DurationTrace],
    real_length: int,
) -> tuple[list[DifferenceRow], dict[tuple[int, str], np.ndarray]]:
    baseline = traces[0]
    rows = []
    token_differences = {}
    for padding, trace in traces.items():
        for stage, tensor in trace.stages.items():
            comparison = compare_real_tokens(
                baseline.stages[stage][0].float().cpu().numpy(),
                tensor[0].float().cpu().numpy(),
                real_length,
            )
            rows.append(
                DifferenceRow(
                    padding,
                    stage,
                    comparison.max_absolute_difference,
                    comparison.mean_absolute_difference,
                )
            )
            token_differences[(padding, stage)] = comparison.per_token_maximum
    return rows, token_differences


def _plot_stage_maximum(rows: list[DifferenceRow], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 7))
    stages = list(dict.fromkeys(row.stage for row in rows))
    for stage in stages:
        selected = [row for row in rows if row.stage == stage]
        axis.plot(
            [row.padding_tokens for row in selected],
            [max(row.max_absolute_difference, 1e-12) for row in selected],
            marker="o",
            label=stage,
        )
    axis.set(
        title="Magda duration path: padding leakage over real tokens",
        xlabel="Right-padding tokens added",
        ylabel="Maximum absolute difference (log scale)",
        yscale="log",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_token_heatmap(
    differences: dict[tuple[int, str], np.ndarray],
    padding: int,
    output: Path,
) -> None:
    stages = [stage for candidate_padding, stage in differences if candidate_padding == padding]
    values = np.vstack(
        [np.maximum(differences[(padding, stage)], 1e-12) for stage in stages]
    )
    figure, axis = plt.subplots(figsize=(14, 7))
    image = axis.imshow(np.log10(values), aspect="auto", interpolation="nearest")
    axis.set(
        title=f"Per-token leakage with +{padding} right-padding tokens",
        xlabel="Real token index",
        ylabel="Duration-path stage",
        yticks=np.arange(len(stages)),
        yticklabels=stages,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log10(max absolute feature difference)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_duration_overlay(
    traces: dict[int, DurationTrace],
    real_length: int,
    padding: int,
    output: Path,
) -> None:
    tokens = np.arange(real_length)
    baseline = traces[0].stages["duration_continuous"][0, :real_length].cpu()
    leaked = traces[padding].stages["duration_continuous"][0, :real_length].cpu()
    fixed = traces[padding].stages["duration_integer_packed_hack"][0, :real_length].cpu()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    top.plot(tokens, baseline, label="no extra padding", linewidth=2)
    top.plot(tokens, leaked, label=f"+{padding} padding, runtime final LSTM", alpha=0.8)
    top.set(ylabel="Continuous predicted duration", title="Duration output comparison")
    top.legend()
    runtime_integer = traces[padding].stages["duration_integer"][0, :real_length].cpu()
    bottom.step(tokens, runtime_integer, where="mid", label="runtime integer duration")
    bottom.step(tokens, fixed, where="mid", label="packed-final-LSTM hack", alpha=0.8)
    bottom.set(xlabel="Real token index", ylabel="Integer predicted duration")
    bottom.legend()
    for axis in (top, bottom):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(
    output_dir: Path,
    traces: dict[int, DurationTrace],
    real_length: int,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, differences = build_rows(traces, real_length)
    with (output_dir / "layer_differences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_stage_maximum(rows, output_dir / "layer_max_difference_vs_padding.png")
    _plot_token_heatmap(differences, 50, output_dir / "token_leakage_plus_50.png")
    _plot_duration_overlay(traces, real_length, 50, output_dir / "durations_plus_50.png")
