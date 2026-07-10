from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    archive_path: Path
    model_path: Path
    results_dir: Path
    seed: int
    voice_count: int
    highlighted_voice_count: int


@dataclass(frozen=True)
class SynthesisProfile:
    name: str
    results_dir: Path
    use_diffusion: bool


@dataclass(frozen=True)
class TextInput:
    text_id: str
    text: str
    input_phoneme_tokens: int


@dataclass(frozen=True)
class ReferenceDuration:
    voice_id: str
    duration_seconds: float


@dataclass(frozen=True)
class RequestMetric:
    voice_id: str
    text_id: str
    text: str
    text_length: int
    input_phoneme_tokens: int
    phoneme_count: int
    predicted_seconds: float
    phonemes_per_second: float
    audio_path: str


@dataclass(frozen=True)
class PhonemeMetric:
    phoneme: str
    duration_seconds: float
    voice_id: str
    text_id: str


@dataclass(frozen=True)
class SummaryRow:
    phoneme: str
    count: int
    average_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    p10_seconds: float
    p90_seconds: float

    @classmethod
    def from_values(cls, phoneme: str, values: list[float]) -> "SummaryRow":
        durations = np.asarray(values, dtype=np.float64)
        return cls(
            phoneme=phoneme,
            count=len(values),
            average_seconds=float(np.mean(durations)),
            minimum_seconds=float(np.min(durations)),
            maximum_seconds=float(np.max(durations)),
            p10_seconds=float(np.percentile(durations, 10)),
            p90_seconds=float(np.percentile(durations, 90)),
        )


def select_names(names: list[str], count: int, seed: int) -> list[str]:
    if len(names) < count:
        raise ValueError(f"Need {count} inputs, found {len(names)}")
    return sorted(Random(seed).sample(sorted(names), count))


def summarize_phonemes(metrics: list[PhonemeMetric]) -> list[SummaryRow]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.phoneme].append(metric.duration_seconds)
    return [
        SummaryRow.from_values(phoneme, grouped[phoneme])
        for phoneme in sorted(grouped)
    ]
