from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import numpy as np
import torch
from tqdm import tqdm

from test_speed.benchmark.benchmark_inference import load_model
from test_speed.solution.aligned_training import (
    AlignedSample,
    aligned_duration_loss,
    load_samples,
    predict,
    validation_error,
)


SOURCE_MODEL = Path("/workspace/converted_models/magda_aligned_duration/model.pth")
OUTPUT_MODEL = Path("/workspace/converted_models/magda_aligned_rate/model.pth")
VOICE_DIR = Path(
    "/workspace/tinfer/test_speed/results/final/magda_original/diffusion/embeddings"
)
RESULTS_DIR = Path(
    "/workspace/tinfer/test_speed/results/evidence/final_training"
)
STANDARD_MANIFEST = Path(
    "/workspace/tinfer/test_speed/results/final/magda_original/diffusion/manifest.json"
)
BALANCED_MANIFEST = Path(
    "/workspace/tinfer/test_speed/results/final/magda_rate/independent/manifest.json"
)
SEED = 20260711
FRAMES_PER_SECOND = 40.0
TRAINING_STEPS = 800


@dataclass(frozen=True)
class RateTrainingMetrics:
    steps: int
    aligned_baseline_log_mae: float
    aligned_tuned_log_mae: float
    best_step: int
    nested_short_slope: float
    nested_long_slope: float
    balanced_short_slope: float
    balanced_long_slope: float
    rate_weight: float


def rate_variance_loss(
    predictions: torch.Tensor,
    spoken_masks: list[torch.Tensor],
) -> torch.Tensor:
    means = []
    for prediction, mask in zip(predictions, spoken_masks):
        selected = prediction[1 : mask.numel() + 1][mask.to(prediction.device)]
        means.append(torch.log(selected.mean()))
    values = torch.stack(means)
    return torch.mean((values - values.mean()) ** 2)


def spoken_mask(sample: AlignedSample, phonemizer) -> torch.Tensor:
    return torch.tensor(
        [
            bool(phonemizer.index_to_symbol[token].strip())
            for token in sample.tokens[1:]
        ],
        dtype=torch.bool,
    )


def phonemes_per_second(
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    durations = prediction[1 : mask.numel() + 1][mask.to(prediction.device)]
    return float(mask.sum() / durations.sum()) * FRAMES_PER_SECOND


def load_rate_samples(path: Path, phonemizer) -> list[AlignedSample]:
    texts = json.loads(path.read_text())["texts"]
    samples = []
    for row in texts:
        phonemes, _ = phonemizer.process_text_with_original_spans(row["text"])
        samples.append(
            AlignedSample(
                row["text_id"],
                [0, *phonemizer.tokenize(phonemes)],
                [],
            )
        )
    return samples


@torch.no_grad()
def controlled_slopes(model, samples, styles, phonemizer) -> tuple[float, float]:
    rates = []
    tokens = []
    for style in styles:
        selected_styles = style.expand(len(samples), -1)
        predictions = predict(model, samples, selected_styles)
        for sample, prediction in zip(samples, predictions):
            mask = spoken_mask(sample, phonemizer).to(prediction.device)
            rates.append(phonemes_per_second(prediction, mask))
            tokens.append(len(sample.tokens))
    x_values = np.asarray(tokens)
    y_values = np.asarray(rates)
    short = np.polyfit(x_values[x_values < 100], y_values[x_values < 100], 1)[0]
    long = np.polyfit(x_values[x_values >= 100], y_values[x_values >= 100], 1)[0]
    return float(short), float(long)


def save_checkpoint(model) -> None:
    checkpoint = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=True)
    checkpoint["net"]["bert_encoder"] = {
        key: value.detach().cpu()
        for key, value in model.bert_encoder.state_dict().items()
    }
    checkpoint["net"]["predictor"] = {
        key: value.detach().cpu()
        for key, value in model.predictor.state_dict().items()
    }
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT_MODEL)


def main() -> None:
    randomizer = random.Random(SEED)
    torch.manual_seed(SEED)
    samples = load_samples()
    randomizer.shuffle(samples)
    validation_count = max(24, len(samples) // 5)
    validation = samples[:validation_count]
    training = samples[validation_count:]
    wrapper = load_model(SOURCE_MODEL, "cuda", "torch")
    model = wrapper._model
    for module in model.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    for parameter in [*model.bert_encoder.parameters(), *model.predictor.parameters()]:
        parameter.requires_grad_(True)
    styles = torch.stack(
        [
            torch.load(path, map_location="cuda", weights_only=True)["voice_vector"]
            for path in sorted(VOICE_DIR.glob("*.pth"))
        ]
    )
    phonemizer = wrapper._get_phonemizer("pl")
    standard_samples = load_rate_samples(STANDARD_MANIFEST, phonemizer)
    balanced_samples = load_rate_samples(BALANCED_MANIFEST, phonemizer)
    baseline = validation_error(model, validation, styles)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert_encoder.parameters(), "lr": 3e-6},
            {"params": model.predictor.parameters(), "lr": 6e-6},
        ],
        weight_decay=1e-4,
    )
    rate_weight = 500.0
    best_score = float("inf")
    best_step = 0
    best_state = None
    best_slopes = (float("inf"),) * 4
    model.bert_encoder.train()
    model.predictor.train()
    for step in tqdm(
        range(1, TRAINING_STEPS + 1),
        desc="Train aligned rate",
        unit="step",
    ):
        actual = randomizer.sample(training, 12)
        standard_rate = randomizer.sample(standard_samples, 16)
        balanced_rate = randomizer.sample(balanced_samples, 16)
        actual_styles = torch.stack(
            [styles[randomizer.randrange(len(styles))] for _ in actual]
        ).squeeze(1)
        rate_style = styles[randomizer.randrange(len(styles))]
        actual_predictions = predict(model, actual, actual_styles)
        standard_predictions = predict(
            model,
            standard_rate,
            rate_style.expand(len(standard_rate), -1),
        )
        balanced_predictions = predict(
            model,
            balanced_rate,
            rate_style.expand(len(balanced_rate), -1),
        )
        aligned_loss = aligned_duration_loss(
            actual_predictions,
            [torch.tensor(sample.durations) for sample in actual],
        )
        rate_loss = 0.5 * rate_variance_loss(
            standard_predictions,
            [spoken_mask(sample, phonemizer) for sample in standard_rate],
        ) + 0.5 * rate_variance_loss(
            balanced_predictions,
            [spoken_mask(sample, phonemizer) for sample in balanced_rate],
        )
        loss = aligned_loss + rate_weight * rate_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*model.bert_encoder.parameters(), *model.predictor.parameters()],
            1.0,
        )
        optimizer.step()
        if step % 25 == 0:
            model.bert_encoder.eval()
            model.predictor.eval()
            error = validation_error(model, validation, styles)
            nested_slopes = controlled_slopes(
                model,
                standard_samples,
                styles,
                phonemizer,
            )
            balanced_slopes = controlled_slopes(
                model,
                balanced_samples,
                styles,
                phonemizer,
            )
            slopes = (*nested_slopes, *balanced_slopes)
            model.bert_encoder.train()
            model.predictor.train()
            score = sum(abs(slope) for slope in slopes)
            if error <= 0.397249698638916 and score < best_score:
                best_score = score
                best_step = step
                best_slopes = slopes
                best_state = (
                    deepcopy(model.bert_encoder.state_dict()),
                    deepcopy(model.predictor.state_dict()),
                )
    assert best_state is not None
    model.bert_encoder.load_state_dict(best_state[0])
    model.predictor.load_state_dict(best_state[1])
    model.bert_encoder.eval()
    model.predictor.eval()
    tuned = validation_error(model, validation, styles)
    save_checkpoint(model)
    metrics = RateTrainingMetrics(
        TRAINING_STEPS,
        baseline,
        tuned,
        best_step,
        best_slopes[0],
        best_slopes[1],
        best_slopes[2],
        best_slopes[3],
        rate_weight,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2) + "\n"
    )
    print(json.dumps(asdict(metrics), indent=2))
    print(f"Checkpoint: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
