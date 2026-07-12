from dataclasses import asdict, dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from test_speed.investigations.bert_context.experiment import (
    BENCHMARK_DIR,
    FRAME_SECONDS,
    LENGTHS,
    MODEL_PATH,
    rotated_passages,
)
from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.padding.analysis import (
    trace_duration_path,
    trace_from_duration_encoded,
    trace_predictor_from_encoded,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "test_speed/results/evidence/duration_path"


@dataclass(frozen=True)
class LocalizationRow:
    voice_id: str
    passage_id: int
    input_phoneme_tokens: int
    condition: str
    phonemes_per_second: float


def rate(
    durations: torch.Tensor,
    spoken: torch.Tensor,
    left_boundary: int = 0,
    right_boundary: int = 0,
) -> float:
    selected = durations[0, : spoken.numel()][spoken]
    assert selected.numel() > left_boundary + right_boundary
    end = selected.numel() - right_boundary
    selected = selected[left_boundary:end]
    return float(selected.numel() / (selected.sum().item() * FRAME_SECONDS))


def regional_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    output = {}
    for label, selected in (
        ("<100", frame[frame["input_phoneme_tokens"] < 100]),
        (">=100", frame[frame["input_phoneme_tokens"] >= 100]),
    ):
        x_values = selected["input_phoneme_tokens"].to_numpy()
        y_values = selected["phonemes_per_second"].to_numpy()
        output[label] = {
            "slope": float(np.polyfit(x_values, y_values, 1)[0]),
            "correlation": float(np.corrcoef(x_values, y_values)[0, 1]),
            "mean": float(y_values.mean()),
            "count": int(y_values.size),
        }
    return output


def plot_rates(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for axis, label, selected in (
        (axes[0], "<100", frame[frame["input_phoneme_tokens"] < 100]),
        (axes[1], ">=100", frame[frame["input_phoneme_tokens"] >= 100]),
    ):
        for condition, group in selected.groupby("condition"):
            means = group.groupby("input_phoneme_tokens")["phonemes_per_second"].mean()
            fit = np.polyfit(means.index, means.values, 1)
            axis.plot(means.index, means.values, marker="o", markersize=3, label=f"{condition}: {fit[0]:.4f}")
        axis.set(title=label, xlabel="Input phoneme tokens", ylabel="Predicted phonemes/s")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "stage_interventions.png", dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    torch.manual_seed(20260711)
    model = load_model(MODEL_PATH, "cuda", "torch")
    for module in model._model.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
    phonemizer = model._get_phonemizer("pl")
    voices = sorted((BENCHMARK_DIR / "embeddings").glob("*.pth"))[:5]
    rows = []
    cases = [(voice, index, text) for voice in voices for index, text in enumerate(rotated_passages())]
    for voice_path, passage_id, text in tqdm(cases, desc="Localize duration mismatch", unit="case"):
        style = torch.load(voice_path, map_location="cuda", weights_only=True)["voice_vector"].reshape(1, 256)
        phonemes, _ = phonemizer.process_text_with_original_spans(text)
        content = phonemizer.tokenize(phonemes)[:510]
        full_tokens = torch.tensor([[0, *content]], device="cuda")
        full = trace_duration_path(model._model, full_tokens, 511, style)
        for length in LENGTHS:
            tokens = full_tokens[:, :length]
            actual = trace_duration_path(model._model, tokens, length, style)
            eos_tokens = torch.cat(
                (tokens, torch.zeros((1, 1), dtype=tokens.dtype, device="cuda")),
                dim=1,
            )
            with_eos = trace_duration_path(
                model._model,
                eos_tokens,
                length + 1,
                style,
            )
            symbols = [phonemizer.index_to_symbol[int(token)] for token in tokens[0]]
            spoken = torch.tensor([bool(symbol.strip()) for symbol in symbols], device="cuda")
            bert_context = trace_predictor_from_encoded(
                model._model,
                full.stages["bert_encoder"][:, :length],
                style,
            )
            text_encoder_context = trace_from_duration_encoded(
                model._model,
                full.stages["predictor.text_encoder.output"][:, :length],
            )
            conditions = {
                "short_actual": actual.stages["duration_integer"],
                "short_with_eos_crop": with_eos.stages["duration_integer"][:, :length],
                "long_through_bert": bert_context.stages["duration_integer"],
                "long_through_text_encoder": text_encoder_context.stages["duration_integer"],
                "long_full_path_crop": full.stages["duration_integer"][:, :length],
            }
            for condition, durations in conditions.items():
                rows.append(LocalizationRow(voice_path.stem, passage_id, length, condition, rate(durations, spoken)))
            if int(spoken.sum()) > 8:
                rows.append(LocalizationRow(voice_path.stem, passage_id, length, "short_drop_first_4", rate(actual.stages["duration_integer"], spoken, 4, 0)))
                rows.append(LocalizationRow(voice_path.stem, passage_id, length, "short_drop_last_4", rate(actual.stages["duration_integer"], spoken, 0, 4)))
                rows.append(LocalizationRow(voice_path.stem, passage_id, length, "short_drop_both_4", rate(actual.stages["duration_integer"], spoken, 4, 4)))
                rows.append(LocalizationRow(voice_path.stem, passage_id, length, "long_full_drop_both_4", rate(full.stages["duration_integer"][:, :length], spoken, 4, 4)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_csv(OUTPUT_DIR / "stage_interventions.csv", index=False)
    metrics = {condition: regional_metrics(group) for condition, group in frame.groupby("condition")}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_rates(frame)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
