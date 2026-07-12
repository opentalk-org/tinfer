from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from test_speed.benchmark.benchmark_inference import load_model
from test_speed.benchmark.benchmark_corpus import POLISH_PASSAGE
from test_speed.investigations.padding.analysis import (
    DurationTrace,
    trace_duration_path,
    trace_from_duration_encoded,
    trace_predictor_from_encoded,
)
from test_speed.investigations.padding.run import _extract_reference
from test_speed.investigations.padding.local_attention_patch import install_local_attention


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "test_speed/results/evidence/genuine_length"
POINT_COUNT = 48
FRAME_SECONDS = 300 / 24_000
LOCAL_RADII = (8, 16, 32)


@dataclass(frozen=True)
class RatePoint:
    token_count: int
    actual: float
    corrected_bert_norm: float
    corrected_bert_direction: float
    local_8: float
    local_16: float
    local_32: float
    frozen_bert: float
    frozen_text_encoder: float
    frozen_final_lstm: float


def _boundary_slice(tensor: torch.Tensor, content_length: int) -> torch.Tensor:
    return torch.cat(
        [tensor[:, : content_length + 1], tensor[:, -1:]],
        dim=1,
    )


def _rate(trace: DurationTrace, symbols: list[str]) -> float:
    durations = trace.stages["duration_integer"][0, 1:].detach().cpu().numpy()
    spoken = np.asarray([bool(symbol.strip()) for symbol in symbols])
    return float(spoken.sum() / (durations[spoken].sum() * FRAME_SECONDS))


def _relative_difference(short: torch.Tensor, long: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(short.float() - long.float())
    scale = torch.linalg.vector_norm(long.float())
    return float((difference / scale).cpu())


def _replace_norms(vectors: torch.Tensor, norm_source: torch.Tensor) -> torch.Tensor:
    source_norms = torch.linalg.vector_norm(norm_source, dim=-1, keepdim=True)
    vector_norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors * (source_norms / vector_norms)


def _plot_rates(points: list[RatePoint]) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    x_values = [point.token_count for point in points]
    for field, label in (
        ("actual", "Actual model"),
        ("corrected_bert_norm", "Only BERT norms replaced"),
        ("corrected_bert_direction", "Only BERT directions replaced"),
        ("frozen_bert", "Long-context BERT frozen"),
        ("frozen_text_encoder", "Long-context text encoder frozen"),
        ("frozen_final_lstm", "Long-context final LSTM frozen"),
    ):
        axis.plot(x_values, [getattr(point, field) for point in points], label=label)
    axis.set(
        title="Magda genuine-length rate counterfactuals",
        xlabel="Input phoneme tokens",
        ylabel="Predicted phonemes/s",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rate_counterfactuals.png", dpi=180)
    plt.close(figure)


def _plot_stage_differences(
    lengths: list[int],
    traces: dict[int, DurationTrace],
    longest: DurationTrace,
) -> None:
    stages = (
        "bert",
        "bert_encoder",
        "predictor.text_encoder.bilstm_1",
        "predictor.text_encoder.bilstm_2",
        "predictor.text_encoder.bilstm_3",
        "predictor.text_encoder.output",
        "predictor.lstm",
        "duration_proj_logits",
        "duration_continuous",
    )
    figure, axis = plt.subplots(figsize=(12, 7))
    for stage in stages:
        values = []
        for length in lengths:
            shared = length - 1
            short = traces[length].stages[stage][:, :shared]
            reference = longest.stages[stage][:, :shared]
            values.append(max(_relative_difference(short, reference), 1e-8))
        axis.plot(lengths, values, label=stage)
    axis.set(
        title="Shared-prefix representation change vs longest real sequence",
        xlabel="Input phoneme tokens",
        ylabel="Relative L2 difference (log scale)",
        yscale="log",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "shared_prefix_stage_differences.png", dpi=180)
    plt.close(figure)


def _plot_local_attention(points: list[RatePoint]) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    x_values = [point.token_count for point in points]
    axis.plot(x_values, [point.actual for point in points], label="Global attention")
    for radius in LOCAL_RADII:
        field = f"local_{radius}"
        axis.plot(
            x_values,
            [getattr(point, field) for point in points],
            label=f"Local attention ±{radius}",
        )
    axis.set(
        title="Magda duration rate with local PLBERT attention",
        xlabel="Input phoneme tokens",
        ylabel="Predicted phonemes/s",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "local_attention_rates.png", dpi=180)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(20260711)
    model = load_model(Path("/workspace/converted_models/magda/model.pth"), "cuda", "torch")
    for module in model._model.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
    reference = _extract_reference()
    model.load_voice_from_audio("magda", str(reference))
    style = model.get_voice("magda").detach().reshape(1, 256)
    phonemizer = model._get_phonemizer("pl")
    phonemes, _ = phonemizer.process_text_with_original_spans(POLISH_PASSAGE)
    all_symbols = list(phonemes)
    all_tokens = phonemizer.tokenize(phonemes)
    content_tokens = all_tokens[:509]
    content_symbols = all_symbols[:509]
    period_token = phonemizer.word_index_dictionary["."]
    lengths = np.linspace(8, 511, POINT_COUNT, dtype=int).tolist()
    traces = {}
    sequences = {}
    for length in lengths:
        content_length = length - 2
        tokens = [0, *content_tokens[:content_length], period_token]
        sequences[length] = tokens
        tensor = torch.tensor([tokens], device="cuda")
        traces[length] = trace_duration_path(model._model, tensor, length, style)
    longest = traces[lengths[-1]]
    install_local_attention(model._model.bert, LOCAL_RADII[0])
    local_traces = {}
    for radius in LOCAL_RADII:
        model._model.bert.local_attention_radius = radius
        local_traces[radius] = {}
        for length in lengths:
            tensor = torch.tensor([sequences[length]], device="cuda")
            local_traces[radius][length] = trace_duration_path(
                model._model, tensor, length, style
            )
    points = []
    for length in lengths:
        content_length = length - 2
        symbols = content_symbols[:content_length] + ["."]
        frozen_bert_input = _boundary_slice(
            longest.stages["bert_encoder"], content_length
        )
        actual_bert_input = traces[length].stages["bert_encoder"]
        corrected_norm = trace_predictor_from_encoded(
            model._model,
            _replace_norms(actual_bert_input, frozen_bert_input),
            style,
        )
        corrected_direction = trace_predictor_from_encoded(
            model._model,
            _replace_norms(frozen_bert_input, actual_bert_input),
            style,
        )
        frozen_bert = trace_predictor_from_encoded(
            model._model, frozen_bert_input, style
        )
        frozen_text_input = _boundary_slice(
            longest.stages["predictor.text_encoder.output"], content_length
        )
        frozen_text = trace_from_duration_encoded(model._model, frozen_text_input)
        frozen_final = DurationTrace(
            {
                "duration_integer": torch.round(
                    torch.sigmoid(
                        model._model.predictor.duration_proj(
                            _boundary_slice(
                                longest.stages["predictor.lstm"], content_length
                            )
                        )
                    ).sum(axis=-1)
                ).clamp(min=1)
            }
        )
        points.append(
            RatePoint(
                length,
                _rate(traces[length], symbols),
                _rate(corrected_norm, symbols),
                _rate(corrected_direction, symbols),
                _rate(local_traces[8][length], symbols),
                _rate(local_traces[16][length], symbols),
                _rate(local_traces[32][length], symbols),
                _rate(frozen_bert, symbols),
                _rate(frozen_text, symbols),
                _rate(frozen_final, symbols),
            )
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_rates(points)
    _plot_local_attention(points)
    _plot_stage_differences(lengths, traces, longest)
    rows = [
        "tokens,actual,corrected_bert_norm,corrected_bert_direction,"
        "local_8,local_16,local_32,"
        "frozen_bert,frozen_text_encoder,frozen_final_lstm"
    ]
    rows.extend(
        f"{p.token_count},{p.actual},{p.corrected_bert_norm},"
        f"{p.corrected_bert_direction},{p.local_8},{p.local_16},{p.local_32},"
        f"{p.frozen_bert},"
        f"{p.frozen_text_encoder},{p.frozen_final_lstm}"
        for p in points
    )
    (OUTPUT_DIR / "rates.csv").write_text("\n".join(rows) + "\n")
    print(f"Genuine-length report: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
