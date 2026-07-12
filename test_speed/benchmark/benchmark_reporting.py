from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from test_speed.benchmark.benchmark_data import (
    PhonemeMetric,
    ReferenceDuration,
    RequestMetric,
    SummaryRow,
    summarize_phonemes,
)
from test_speed.benchmark.benchmark_style import StyleEmbeddingNorm, write_style_norm_plots


def _write_dataclass_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty metrics table: {path.name}")
    serialized = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(serialized[0]))
        writer.writeheader()
        writer.writerows(serialized)


def _write_dataclass_json(path: Path, rows: list[object]) -> None:
    serialized = [asdict(row) for row in rows]
    path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_raw_metrics(
    results_dir: Path,
    requests: list[RequestMetric],
    phonemes: list[PhonemeMetric],
) -> None:
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(metrics_dir / "requests.csv", requests)
    _write_dataclass_json(metrics_dir / "requests.json", requests)
    _write_dataclass_csv(metrics_dir / "phoneme_occurrences.csv", phonemes)
    _write_dataclass_json(metrics_dir / "phoneme_occurrences.json", phonemes)


def write_summary_table(path: Path, rows: list[SummaryRow]) -> None:
    _write_dataclass_csv(path, rows)


def plot_scatter(
    metrics: list[RequestMetric],
    title: str,
    path: Path,
) -> None:
    if not metrics:
        raise ValueError(f"Cannot plot empty request metrics: {title}")
    figure, axis = plt.subplots(figsize=(10, 6))
    x_values, y_values = scatter_coordinates(metrics)
    axis.scatter(
        x_values,
        y_values,
        alpha=0.65,
    )
    axis.set(
        title=title,
        xlabel="Input phoneme tokens",
        ylabel="Predicted phonemes/s",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def scatter_coordinates(
    metrics: list[RequestMetric],
) -> tuple[list[int], list[float]]:
    return (
        [row.input_phoneme_tokens for row in metrics],
        [row.phonemes_per_second for row in metrics],
    )


def reference_all_run_coordinates(
    metrics: list[RequestMetric],
    references: list[ReferenceDuration],
) -> tuple[list[float], list[float]]:
    durations = {item.voice_id: item.duration_seconds for item in references}
    return (
        [durations[row.voice_id] for row in metrics],
        [row.phonemes_per_second for row in metrics],
    )


def reference_mean_voice_coordinates(
    metrics: list[RequestMetric],
    references: list[ReferenceDuration],
) -> tuple[list[float], list[float]]:
    durations = {item.voice_id: item.duration_seconds for item in references}
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.voice_id].append(metric.phonemes_per_second)
    voice_ids = sorted(grouped)
    return (
        [durations[voice_id] for voice_id in voice_ids],
        [float(np.mean(grouped[voice_id])) for voice_id in voice_ids],
    )


def plot_reference_scatter(
    coordinates: tuple[list[float], list[float]],
    title: str,
    path: Path,
) -> None:
    x_values, y_values = coordinates
    if not x_values:
        raise ValueError(f"Cannot plot empty reference metrics: {title}")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(x_values, y_values, alpha=0.65)
    axis.set(
        title=title,
        xlabel="Reference length (seconds)",
        ylabel="Predicted phonemes/s",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def mean_rates_by_voice(metrics: list[RequestMetric]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.voice_id].append(metric.phonemes_per_second)
    return [
        float(np.mean(grouped[voice_id]))
        for voice_id in sorted(grouped)
    ]


def shared_histogram_edges(
    *groups: list[float],
    bin_width: float,
) -> np.ndarray:
    if bin_width <= 0:
        raise ValueError("Histogram bin width must be positive")
    values = np.asarray(
        [value for group in groups for value in group],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("Histogram values cannot be empty")
    lower = float(np.floor(values.min() / bin_width) * bin_width)
    upper = float(np.ceil(values.max() / bin_width) * bin_width)
    upper = max(upper, lower + bin_width)
    return np.arange(lower, upper + bin_width, bin_width)


def plot_histogram(
    values: list[float],
    edges: np.ndarray,
    title: str,
    population_label: str,
    path: Path,
) -> None:
    if not values:
        raise ValueError(f"Cannot plot empty histogram: {title}")
    figure, axis = plt.subplots(figsize=(10, 6))
    counts, _, _ = axis.hist(values, bins=edges, edgecolor="black", alpha=0.75)
    assert int(counts.sum()) == len(values), "histogram population count mismatch"
    axis.set(
        title=title,
        xlabel="Predicted phonemes/s",
        ylabel=population_label,
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary_index(
    summary_dir: Path,
    voice_ids: list[str],
    has_reference_durations: bool,
) -> Path:
    reference_lines = [
        "- [Reference length vs phonemes/s, all runs]"
        "(reference_duration_vs_phonemes_per_second_all_runs.png)",
        "- [Reference length vs mean phonemes/s by voice]"
        "(reference_duration_vs_mean_phonemes_per_second_by_voice.png)",
    ] if has_reference_durations else []
    lines = [
        "# Phoneme Duration Benchmark",
        "",
        "## Global",
        "",
        "- [Phoneme duration table](global_phoneme_durations.csv)",
        "- [Phonemes/s scatter](global_phonemes_per_second.png)",
        *reference_lines,
        "- [Full style norm vs input tokens](style_embedding_full_norm_vs_input_tokens.png)",
        "- [First 128 style norm vs input tokens](style_embedding_first_128_norm_vs_input_tokens.png)",
        "- [Second 128 style norm vs input tokens](style_embedding_second_128_norm_vs_input_tokens.png)",
        "- [Phonemes/s by voice histogram](phonemes_per_second_by_voice.png)",
        "- [Phonemes/s all runs histogram](phonemes_per_second_all_runs.png)",
        "",
        "## Highlighted voices",
        "",
    ]
    for voice_id in voice_ids:
        lines.extend(
            [
                f"### {voice_id}",
                "",
                f"- [Phoneme duration table]({voice_id}_phoneme_durations.csv)",
                f"- [Phonemes/s scatter]({voice_id}_phonemes_per_second.png)",
                "",
            ]
        )
    index_path = summary_dir / "README.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path

def write_reports(
    results_dir: Path,
    requests: list[RequestMetric],
    phonemes: list[PhonemeMetric],
    references: list[ReferenceDuration],
    style_norms: list[StyleEmbeddingNorm],
    highlighted_voice_ids: list[str],
    histogram_edges: np.ndarray,
) -> Path:
    write_raw_metrics(results_dir, requests, phonemes)
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    scopes = [("global", requests, phonemes)]
    scopes.extend(
        (
            voice_id,
            [row for row in requests if row.voice_id == voice_id],
            [row for row in phonemes if row.voice_id == voice_id],
        )
        for voice_id in highlighted_voice_ids
    )
    for scope, scope_requests, scope_phonemes in tqdm(
        scopes,
        desc="Write reports",
        unit="report",
    ):
        write_summary_table(
            summary_dir / f"{scope}_phoneme_durations.csv",
            summarize_phonemes(scope_phonemes),
        )
        plot_scatter(
            scope_requests,
            f"Predictor rate: {scope}",
            summary_dir / f"{scope}_phonemes_per_second.png",
        )
    plot_histogram(
        mean_rates_by_voice(requests),
        histogram_edges,
        "Mean predictor rate by voice",
        "Number of voices",
        summary_dir / "phonemes_per_second_by_voice.png",
    )
    plot_histogram(
        [row.phonemes_per_second for row in requests],
        histogram_edges,
        "Predictor rate across all voice/text runs",
        "Number of runs",
        summary_dir / "phonemes_per_second_all_runs.png",
    )
    if references:
        plot_reference_scatter(
            reference_all_run_coordinates(requests, references),
            "Reference length vs predictor rate: all runs",
            summary_dir / "reference_duration_vs_phonemes_per_second_all_runs.png",
        )
        plot_reference_scatter(
            reference_mean_voice_coordinates(requests, references),
            "Reference length vs mean predictor rate by voice",
            summary_dir
            / "reference_duration_vs_mean_phonemes_per_second_by_voice.png",
        )
    write_style_norm_plots(requests, style_norms, summary_dir)
    return _write_summary_index(
        summary_dir,
        highlighted_voice_ids,
        bool(references),
    )
