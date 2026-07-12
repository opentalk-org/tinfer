from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.nn import functional as F

from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.duration_training.corpus import (
    build_context_pair,
    concatenate_samples,
    fetch_backend_segments,
    tokenize_segments,
)
from test_speed.investigations.duration_training.duration_model import (
    duration_forward,
    make_token_batch,
)
from test_speed.investigations.duration_training.evaluation import evaluate
from test_speed.investigations.padding.run import _extract_reference


SOURCE_MODEL = Path("/workspace/converted_models/magda/model.pth")
TUNED_MODEL = Path(
    "/workspace/converted_models/experiments/magda_duration_consistency/model.pth"
)
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results/evidence/duration_training/checkpoint_audit.json"
)


@torch.no_grad()
def predict_batch(model, style, sequences):
    batch = make_token_batch(sequences, str(style.device))
    bert = model.bert(batch.tokens, attention_mask=(~batch.mask).int())
    durations = duration_forward(
        model.bert_encoder,
        model.predictor,
        bert,
        style.expand(len(sequences), -1),
        batch,
    )
    return durations, batch.lengths


def centered_values(durations, lengths):
    values = []
    for row, length in zip(durations, lengths.tolist()):
        selected = torch.log(row[1 : length - 1])
        values.append(selected - selected.mean())
    return torch.cat(values).cpu().numpy()


def context_error(model, style, pairs):
    short, _ = predict_batch(model, style, [pair.short_tokens for pair in pairs])
    full, _ = predict_batch(model, style, [pair.long_tokens for pair in pairs])
    errors = []
    for short_row, full_row, pair in zip(short, full, pairs):
        end = pair.shared_length - 4
        errors.append(
            F.l1_loss(torch.log(short_row[1:end]), torch.log(full_row[1:end]))
        )
    return float(torch.stack(errors).mean())


def changed_modules():
    source = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=True)["net"]
    tuned = torch.load(TUNED_MODEL, map_location="cpu", weights_only=True)["net"]
    return [
        module
        for module in source
        if not all(
            torch.equal(source[module][key], tuned[module][key])
            for key in source[module]
        )
    ]


def main() -> None:
    source = load_model(SOURCE_MODEL, "cuda", "torch")
    tuned = load_model(TUNED_MODEL, "cuda", "torch")
    reference = _extract_reference()
    source.load_voice_from_audio("magda", str(reference))
    style_array = source.get_voice("magda").detach().cpu().numpy().copy()
    style = torch.from_numpy(style_array).cuda().reshape(1, 256)
    phonemizer = source._get_phonemizer("pl")
    samples = [
        sample
        for sample in tokenize_segments(fetch_backend_segments(), phonemizer)
        if 7 < len(sample.tokens) <= 511 and sample.duration_seconds > 0
    ]
    random.Random(20260711).shuffle(samples)
    validation = samples[: max(24, len(samples) // 5)]
    source_duration, lengths = predict_batch(
        source._model, style, [sample.tokens for sample in validation]
    )
    tuned_duration, _ = predict_batch(
        tuned._model, style, [sample.tokens for sample in validation]
    )
    shape_correlation = float(
        np.corrcoef(
            centered_values(source_duration, lengths),
            centered_values(tuned_duration, lengths),
        )[0, 1]
    )
    randomizer = random.Random(991)
    terminal = phonemizer.word_index_dictionary["."]
    pairs = []
    for _ in range(32):
        combined = concatenate_samples(
            validation, randomizer.randint(160, 511), terminal, randomizer
        )
        short_length = randomizer.randint(8, min(128, len(combined.tokens) - 2))
        pairs.append(build_context_pair(combined.tokens, short_length, terminal))
    independent_metrics, _ = evaluate(
        tuned._model, style, phonemizer, validation
    )
    audit = {
        "changed_modules": changed_modules(),
        "independent_reload": independent_metrics.__dict__,
        "heldout_context_log_mae_baseline": context_error(
            source._model, style, pairs
        ),
        "heldout_context_log_mae_tuned": context_error(
            tuned._model, style, pairs
        ),
        "centered_log_duration_correlation": shape_correlation,
        "audio_decoder_used": False,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
