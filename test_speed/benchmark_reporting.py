from dataclasses import asdict
from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from test_speed.benchmark_data import (
    PhonemeMetric,
    RequestMetric,
    SummaryRow,
    summarize_phonemes,
)


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
    axis.scatter(
        [row.text_length for row in metrics],
        [row.phonemes_per_second for row in metrics],
        alpha=0.65,
    )
    axis.set(
        title=title,
        xlabel="Input text length (characters)",
        ylabel="Predicted phonemes/s",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary_index(summary_dir: Path, voice_ids: list[str]) -> Path:
    lines = [
        "# Phoneme Duration Benchmark",
        "",
        "## Global",
        "",
        "- [Phoneme duration table](global_phoneme_durations.csv)",
        "- [Phonemes/s scatter](global_phonemes_per_second.png)",
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
    highlighted_voice_ids: list[str],
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
    return _write_summary_index(summary_dir, highlighted_voice_ids)
