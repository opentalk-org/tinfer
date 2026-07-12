from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import torch
from torch.nn import functional as F
from tqdm import tqdm

from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.duration_training.duration_model import duration_forward, make_token_batch


SOURCE_MODEL = Path("/workspace/converted_models/magda/model.pth")
OUTPUT_MODEL = Path("/workspace/converted_models/magda_aligned_duration/model.pth")
ALIGNMENTS_PATH = Path(
    "/workspace/tinfer/test_speed/results/evidence/alignments/"
    "aligned_durations.json"
)
VOICE_DIR = Path(
    "/workspace/tinfer/test_speed/results/final/magda_original/diffusion/embeddings"
)
RESULTS_DIR = Path(
    "/workspace/tinfer/test_speed/results/evidence/aligned_training"
)
SEED = 20260711


@dataclass(frozen=True)
class AlignedSample:
    segment_id: str
    tokens: list[int]
    durations: list[float]


@dataclass(frozen=True)
class TrainingMetrics:
    steps: int
    training_samples: int
    validation_samples: int
    baseline_validation_log_mae: float
    tuned_validation_log_mae: float
    best_step: int
    target_frame_seconds: float
    trainable_modules: list[str]


def aligned_duration_loss(
    predictions: torch.Tensor,
    targets: list[torch.Tensor],
) -> torch.Tensor:
    losses = []
    for prediction, target in zip(predictions, targets):
        selected = prediction[1 : target.numel() + 1]
        losses.append(
            F.smooth_l1_loss(
                torch.log1p(selected),
                torch.log1p(target.to(prediction.device)),
            )
        )
    return torch.stack(losses).mean()


def load_samples() -> list[AlignedSample]:
    payload = json.loads(ALIGNMENTS_PATH.read_text())
    return [
        AlignedSample(
            row["segment_id"],
            row["tokens"][:-1],
            row["durations"][1:-1],
        )
        for row in payload["segments"]
    ]


def synthetic_sample(
    samples: list[AlignedSample],
    maximum: int,
    randomizer: random.Random,
) -> AlignedSample:
    tokens = [0]
    durations = []
    identifiers = []
    for sample in randomizer.sample(samples, len(samples)):
        available = maximum - len(tokens)
        if available == 0:
            break
        included_tokens = sample.tokens[1 : available + 1]
        included_durations = sample.durations[:available]
        tokens.extend(included_tokens)
        durations.extend(included_durations)
        identifiers.append(sample.segment_id)
    assert len(tokens) == maximum
    return AlignedSample("+".join(identifiers), tokens, durations)


def predict(model, samples: list[AlignedSample], styles: torch.Tensor) -> torch.Tensor:
    batch = make_token_batch([sample.tokens for sample in samples], "cuda")
    with torch.no_grad():
        bert = model.bert(batch.tokens, attention_mask=(~batch.mask).int())
    return duration_forward(
        model.bert_encoder,
        model.predictor,
        bert,
        styles,
        batch,
    )


@torch.no_grad()
def validation_error(model, samples, styles) -> float:
    errors = []
    for start in range(0, len(samples), 16):
        selected = samples[start : start + 16]
        selected_styles = torch.stack(
            [styles[(start + index) % len(styles)] for index in range(len(selected))]
        ).squeeze(1)
        predictions = predict(model, selected, selected_styles)
        for prediction, sample in zip(predictions, selected):
            target = torch.tensor(sample.durations, device="cuda")
            predicted = prediction[1 : len(sample.durations) + 1]
            errors.append(torch.abs(torch.log1p(predicted) - torch.log1p(target)).cpu())
    return float(torch.cat(errors).mean())


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
    baseline = validation_error(model, validation, styles)
    model.bert_encoder.train()
    model.predictor.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert_encoder.parameters(), "lr": 5e-6},
            {"params": model.predictor.parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    best_error = float("inf")
    best_step = 0
    best_state = None
    for step in tqdm(range(1, 401), desc="Train aligned duration", unit="step"):
        actual = randomizer.sample(training, 12)
        synthetic = [
            synthetic_sample(training, randomizer.randint(160, 511), randomizer)
            for _ in range(4)
        ]
        selected = [*actual, *synthetic]
        selected_styles = torch.stack(
            [styles[randomizer.randrange(len(styles))] for _ in selected]
        ).squeeze(1)
        predictions = predict(model, selected, selected_styles)
        targets = [torch.tensor(sample.durations) for sample in selected]
        loss = aligned_duration_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*model.bert_encoder.parameters(), *model.predictor.parameters()],
            1.0,
        )
        optimizer.step()
        if step % 50 == 0:
            model.bert_encoder.eval()
            model.predictor.eval()
            error = validation_error(model, validation, styles)
            model.bert_encoder.train()
            model.predictor.train()
            if error < best_error:
                best_error = error
                best_step = step
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
    metrics = TrainingMetrics(
        400,
        len(training),
        len(validation),
        baseline,
        tuned,
        best_step,
        0.025,
        ["bert_encoder", "predictor"],
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2) + "\n"
    )
    print(json.dumps(asdict(metrics), indent=2))
    print(f"Checkpoint: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
