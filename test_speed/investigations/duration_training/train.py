from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import random

import numpy as np
import torch

from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.duration_training.corpus import (
    build_context_pair,
    concatenate_samples,
    fetch_backend_segments,
    tokenize_segments,
)
from test_speed.investigations.duration_training.duration_model import (
    centered_shape_loss,
    consistency_loss,
    duration_forward,
    make_token_batch,
    total_duration_loss,
)
from test_speed.investigations.duration_training.evaluation import (
    FRAME_SECONDS,
    evaluate,
    write_evaluation,
)
from test_speed.investigations.duration_training.schemas import TrainingConfig, TrainingRecord
from test_speed.investigations.padding.run import _extract_reference


ROOT = Path(__file__).resolve().parents[3]
SOURCE_MODEL = Path("/workspace/converted_models/magda/model.pth")
OUTPUT_MODEL = Path(
    "/workspace/converted_models/experiments/magda_duration_consistency/model.pth"
)
RESULTS_DIR = ROOT / "test_speed/results/evidence/duration_training"

def _bert_output(model, batch):
    with torch.no_grad():
        return model.bert(batch.tokens, attention_mask=(~batch.mask).int())


def _duration_batch(model, style, sequences):
    batch = make_token_batch(sequences, str(style.device))
    bert_output = _bert_output(model, batch)
    durations = duration_forward(
        model.bert_encoder,
        model.predictor,
        bert_output,
        style.expand(len(sequences), -1),
        batch,
    )
    return batch, bert_output, durations


def _save_checkpoint(model) -> None:
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


def _training_samples(phonemizer, seed):
    samples = [
        sample
        for sample in tokenize_segments(fetch_backend_segments(), phonemizer)
        if 7 < len(sample.tokens) <= 511 and sample.duration_seconds > 0
    ]
    random.Random(seed).shuffle(samples)
    validation_count = max(24, len(samples) // 5)
    return samples[validation_count:], samples[:validation_count]


def _synthetic_pairs(samples, config, randomizer, terminal):
    synthetic = []
    pairs = []
    for _ in range(config.synthetic_batch_size):
        maximum = randomizer.randint(160, 511)
        combined = concatenate_samples(samples, maximum, terminal, randomizer)
        short_maximum = min(128, len(combined.tokens) - 2)
        short_length = randomizer.randint(8, short_maximum)
        synthetic.append(combined)
        pairs.append(build_context_pair(combined.tokens, short_length, terminal))
    return synthetic, pairs


def _train_step(
    model,
    teacher_encoder,
    teacher_predictor,
    style,
    samples,
    optimizer,
    config,
    randomizer,
    terminal,
    step,
):
    selected = randomizer.sample(samples, config.actual_batch_size)
    actual_batch, actual_bert, actual_duration = _duration_batch(
        model, style, [sample.tokens for sample in selected]
    )
    target_frames = torch.tensor(
        [sample.duration_seconds / FRAME_SECONDS for sample in selected],
        device=style.device,
    )
    actual_loss = total_duration_loss(
        actual_duration, actual_batch.lengths, target_frames
    )
    with torch.no_grad():
        teacher_duration = duration_forward(
            teacher_encoder,
            teacher_predictor,
            actual_bert,
            style.expand(len(selected), -1),
            actual_batch,
        )
    shape_loss = centered_shape_loss(
        actual_duration, teacher_duration, actual_batch.lengths
    )
    synthetic, pairs = _synthetic_pairs(samples, config, randomizer, terminal)
    full_batch, _, full_duration = _duration_batch(
        model, style, [item.tokens for item in synthetic]
    )
    short_batch, _, short_duration = _duration_batch(
        model, style, [pair.short_tokens for pair in pairs]
    )
    synthetic_targets = torch.tensor(
        [item.duration_seconds / FRAME_SECONDS for item in synthetic],
        device=style.device,
    )
    synthetic_loss = total_duration_loss(
        full_duration, full_batch.lengths, synthetic_targets
    )
    context_loss = consistency_loss(
        short_duration,
        full_duration,
        torch.tensor([pair.shared_length for pair in pairs], device=style.device),
        config.boundary_tokens,
    )
    ramp = min(1.0, step / 100)
    loss = (
        actual_loss
        + synthetic_loss
        + config.consistency_weight * ramp * context_loss
        + config.shape_weight * shape_loss
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [*model.bert_encoder.parameters(), *model.predictor.parameters()], 1.0
    )
    optimizer.step()
    return loss, actual_loss, synthetic_loss, context_loss, shape_loss


def train(config: TrainingConfig) -> None:
    randomizer = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model_wrapper = load_model(SOURCE_MODEL, "cuda", "torch")
    model = model_wrapper._model
    for module in model.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    for parameter in [*model.bert_encoder.parameters(), *model.predictor.parameters()]:
        parameter.requires_grad_(True)
    for module in model.predictor.modules():
        if isinstance(module, torch.nn.LSTM):
            module.train()
    teacher_wrapper = load_model(SOURCE_MODEL, "cuda", "torch")
    teacher_encoder = teacher_wrapper._model.bert_encoder.eval()
    teacher_predictor = teacher_wrapper._model.predictor.eval()
    for parameter in [*teacher_encoder.parameters(), *teacher_predictor.parameters()]:
        parameter.requires_grad_(False)
    reference = _extract_reference()
    model_wrapper.load_voice_from_audio("magda", str(reference))
    style_array = model_wrapper.get_voice("magda").detach().cpu().numpy().copy()
    style = torch.from_numpy(style_array).to("cuda").reshape(1, 256)
    phonemizer = model_wrapper._get_phonemizer("pl")
    training_samples, validation_samples = _training_samples(
        phonemizer, config.seed
    )
    baseline_metrics, baseline_rates = evaluate(
        model, style, phonemizer, validation_samples
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert_encoder.parameters(), "lr": config.encoder_learning_rate},
            {"params": model.predictor.parameters(), "lr": config.predictor_learning_rate},
        ],
        weight_decay=1e-4,
    )
    terminal = phonemizer.word_index_dictionary["."]
    records = []
    best_score = float("inf")
    best_state = None
    for step in range(1, config.steps + 1):
        losses = _train_step(
            model,
            teacher_encoder,
            teacher_predictor,
            style,
            training_samples,
            optimizer,
            config,
            randomizer,
            terminal,
            step,
        )
        evaluation = None
        if step % config.evaluation_interval == 0 or step == config.steps:
            evaluation, _ = evaluate(model, style, phonemizer, validation_samples)
            score = (
                abs(evaluation.rate_drift) / abs(baseline_metrics.rate_drift)
                + evaluation.validation_log_duration_mae
                / baseline_metrics.validation_log_duration_mae
            )
            if score < best_score:
                best_score = score
                best_state = (
                    deepcopy(model.bert_encoder.state_dict()),
                    deepcopy(model.predictor.state_dict()),
                )
            print(
                f"step={step} loss={float(losses[0]):.4f} "
                f"drift={evaluation.rate_drift:.3f} "
                f"val={evaluation.validation_log_duration_mae:.4f}"
            )
        records.append(
            TrainingRecord(
                step,
                *[float(loss.detach()) for loss in losses],
                evaluation.rate_drift if evaluation else None,
                evaluation.validation_log_duration_mae if evaluation else None,
            )
        )
    assert best_state is not None
    model.bert_encoder.load_state_dict(best_state[0])
    model.predictor.load_state_dict(best_state[1])
    tuned_metrics, tuned_rates = evaluate(model, style, phonemizer, validation_samples)
    _save_checkpoint(model)
    metrics = {
        "config": asdict(config),
        "training_samples": len(training_samples),
        "validation_samples": len(validation_samples),
        "baseline": asdict(baseline_metrics),
        "tuned": asdict(tuned_metrics),
        "best_score": best_score,
        "audio_decoder_used": False,
        "trainable_modules": ["bert_encoder", "predictor"],
    }
    write_evaluation(RESULTS_DIR, baseline_rates, tuned_rates, metrics)
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Fine-tuned checkpoint: {OUTPUT_MODEL}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--shape-weight", type=float, default=0.75)
    arguments = parser.parse_args()
    train(
        TrainingConfig(
            steps=arguments.steps,
            shape_weight=arguments.shape_weight,
        )
    )


if __name__ == "__main__":
    main()
