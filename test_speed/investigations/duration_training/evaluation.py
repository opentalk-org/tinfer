from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from test_speed.benchmark.benchmark_corpus import POLISH_PASSAGE
from test_speed.investigations.duration_training.duration_model import (
    duration_forward,
    make_token_batch,
)


FRAME_SECONDS = 300 / 24_000


@dataclass(frozen=True)
class RateResult:
    token_count: int
    phonemes_per_second: float


@dataclass(frozen=True)
class EvaluationResult:
    short_rate: float
    long_rate: float
    rate_drift: float
    validation_log_duration_mae: float


def controlled_sequences(phonemizer, point_count: int) -> list[list[int]]:
    phonemes, _ = phonemizer.process_text_with_original_spans(POLISH_PASSAGE)
    content = phonemizer.tokenize(phonemes)[:509]
    terminal = phonemizer.word_index_dictionary["."]
    lengths = np.linspace(8, 511, point_count, dtype=int)
    return [[0, *content[: length - 2], terminal] for length in lengths]


@torch.no_grad()
def predict_sequences(model, style, sequences) -> list[torch.Tensor]:
    predictions = []
    for tokens in sequences:
        batch = make_token_batch([tokens], str(style.device))
        bert_output = model.bert(
            batch.tokens,
            attention_mask=(~batch.mask).int(),
        )
        duration = duration_forward(
            model.bert_encoder,
            model.predictor,
            bert_output,
            style,
            batch,
        )
        predictions.append(duration[0, : len(tokens)].detach().cpu())
    return predictions


def rate_results(phonemizer, sequences, predictions) -> list[RateResult]:
    results = []
    for tokens, durations in zip(sequences, predictions):
        symbols = [phonemizer.index_to_symbol[token] for token in tokens[1:]]
        spoken = torch.tensor([bool(symbol.strip()) for symbol in symbols])
        seconds = float(durations[1:][spoken].sum()) * FRAME_SECONDS
        results.append(RateResult(len(tokens), int(spoken.sum()) / seconds))
    return results


@torch.no_grad()
def validation_duration_error(model, style, samples) -> float:
    errors = []
    for start in range(0, len(samples), 16):
        selected = samples[start : start + 16]
        batch = make_token_batch([sample.tokens for sample in selected], str(style.device))
        bert_output = model.bert(batch.tokens, attention_mask=(~batch.mask).int())
        expanded_style = style.expand(len(selected), -1)
        durations = duration_forward(
            model.bert_encoder,
            model.predictor,
            bert_output,
            expanded_style,
            batch,
        )
        positions = torch.arange(durations.shape[1], device=durations.device)
        predicted = (
            durations * (positions.unsqueeze(0) < batch.lengths.unsqueeze(1))
        ).sum(dim=1)
        targets = torch.tensor(
            [sample.duration_seconds / FRAME_SECONDS for sample in selected],
            device=durations.device,
        )
        errors.extend(torch.abs(torch.log1p(predicted) - torch.log1p(targets)).cpu())
    return float(torch.stack(errors).mean())


def evaluate(model, style, phonemizer, validation_samples) -> tuple[EvaluationResult, list[RateResult]]:
    sequences = controlled_sequences(phonemizer, 48)
    rates = rate_results(
        phonemizer,
        sequences,
        predict_sequences(model, style, sequences),
    )
    short = float(np.mean([row.phonemes_per_second for row in rates if row.token_count <= 50]))
    long = float(np.mean([row.phonemes_per_second for row in rates if row.token_count >= 400]))
    result = EvaluationResult(
        short,
        long,
        long - short,
        validation_duration_error(model, style, validation_samples),
    )
    return result, rates


def write_evaluation(
    output_dir: Path,
    baseline: list[RateResult],
    tuned: list[RateResult],
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rate_comparison.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("tokens", "baseline_ph_s", "tuned_ph_s"))
        writer.writerows(
            (before.token_count, before.phonemes_per_second, after.phonemes_per_second)
            for before, after in zip(baseline, tuned)
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        [row.token_count for row in baseline],
        [row.phonemes_per_second for row in baseline],
        label="Baseline",
    )
    axis.plot(
        [row.token_count for row in tuned],
        [row.phonemes_per_second for row in tuned],
        label="Duration consistency fine-tune",
    )
    axis.set(
        title="Magda duration-only fine-tuning",
        xlabel="Input phoneme tokens",
        ylabel="Predicted phonemes/s",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "rate_comparison.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(figure)
