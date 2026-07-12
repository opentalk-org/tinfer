from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from test_speed.benchmark.benchmark_corpus import POLISH_PASSAGE
from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.padding.analysis import (
    DurationTrace,
    trace_duration_path,
    trace_predictor_from_encoded,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(
    "/workspace/converted_models/experiments/magda_duration_consistency/model.pth"
)
BENCHMARK_DIR = ROOT / "test_speed/results/evidence/reproduced/duration_training/diffusion"
OUTPUT_DIR = ROOT / "test_speed/results/evidence/bert_context"
FRAME_SECONDS = 300 / 24_000
LENGTHS = (8, 12, 16, 24, 32, 40, 50, 60, 72, 84, 96, 110, 140, 180, 240, 320, 400, 480, 511)
STAGES = (
    "bert",
    "bert_encoder",
    "predictor.text_encoder.bilstm_1",
    "predictor.text_encoder.bilstm_2",
    "predictor.text_encoder.bilstm_3",
    "predictor.text_encoder.output",
    "predictor.lstm",
    "duration_continuous",
)


@dataclass(frozen=True)
class RateRow:
    voice_id: str
    passage_id: int
    input_phoneme_tokens: int
    condition: str
    continuous_phonemes_per_second: float
    integer_phonemes_per_second: float


@dataclass(frozen=True)
class DivergenceRow:
    voice_id: str
    passage_id: int
    input_phoneme_tokens: int
    stage: str
    relative_l2: float
    mean_cosine: float
    relative_norm_change: float
    centered_relative_l2: float


@dataclass(frozen=True)
class CrosscheckRow:
    voice_id: str
    text_id: str
    benchmark_rate: float
    traced_rate: float
    absolute_difference: float


def rotated_passages() -> list[str]:
    words = POLISH_PASSAGE.split()
    offsets = (0, len(words) // 5, 2 * len(words) // 5, 3 * len(words) // 5)
    return [" ".join([*words[offset:], *words[:offset]]) for offset in offsets]


def shared(tensor: torch.Tensor, length: int) -> torch.Tensor:
    return tensor[:, :length]


def replace_norms(vectors: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    vector_norm = torch.linalg.vector_norm(vectors.float(), dim=-1, keepdim=True)
    source_norm = torch.linalg.vector_norm(source.float(), dim=-1, keepdim=True)
    return vectors * (source_norm / vector_norm).to(vectors.dtype)


def rate(trace: DurationTrace, spoken: torch.Tensor, continuous: bool) -> float:
    key = "duration_continuous" if continuous else "duration_integer"
    durations = trace.stages[key][0, : spoken.numel()][spoken]
    return float(spoken.sum().item() / (durations.sum().item() * FRAME_SECONDS))


def compare(short: torch.Tensor, long: torch.Tensor) -> tuple[float, float, float, float]:
    left = short.float().reshape(short.shape[0], short.shape[1], -1)
    right = long.float().reshape(long.shape[0], long.shape[1], -1)
    delta = torch.linalg.vector_norm(left - right)
    scale = torch.linalg.vector_norm(right)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1).mean()
    norms_left = torch.linalg.vector_norm(left, dim=-1)
    norms_right = torch.linalg.vector_norm(right, dim=-1)
    norm_change = torch.linalg.vector_norm(norms_left - norms_right) / torch.linalg.vector_norm(norms_right)
    centered_left = left - left.mean(dim=-1, keepdim=True)
    centered_right = right - right.mean(dim=-1, keepdim=True)
    centered = torch.linalg.vector_norm(centered_left - centered_right) / torch.linalg.vector_norm(centered_right)
    return float(delta / scale), float(cosine), float(norm_change), float(centered)


def main() -> None:
    torch.manual_seed(20260711)
    model = load_model(MODEL_PATH, "cuda", "torch")
    for module in model._model.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
    phonemizer = model._get_phonemizer("pl")
    voice_paths = sorted((BENCHMARK_DIR / "embeddings").glob("*.pth"))[:5]
    rates = []
    divergences = []
    crosschecks = []
    work = [(voice, index, text) for voice in voice_paths for index, text in enumerate(rotated_passages())]
    for voice_path, passage_id, text in tqdm(work, desc="Trace contexts", unit="case"):
        style = torch.load(voice_path, map_location="cuda", weights_only=True)["voice_vector"].reshape(1, 256)
        phonemes, _ = phonemizer.process_text_with_original_spans(text)
        content = phonemizer.tokenize(phonemes)[:510]
        full_tokens = torch.tensor([[0, *content]], device="cuda")
        full_trace = trace_duration_path(model._model, full_tokens, 511, style)
        for length in LENGTHS:
            tokens = full_tokens[:, :length]
            actual = trace_duration_path(model._model, tokens, length, style)
            symbols = [phonemizer.index_to_symbol[int(token)] for token in tokens[0]]
            spoken = torch.tensor([bool(symbol.strip()) for symbol in symbols], device="cuda")
            long_bert = shared(full_trace.stages["bert_encoder"], length)
            short_bert = actual.stages["bert_encoder"]
            conditions = {
                "actual": actual,
                "long_context_bert": trace_predictor_from_encoded(model._model, long_bert, style),
                "long_direction_short_norm": trace_predictor_from_encoded(model._model, replace_norms(long_bert, short_bert), style),
                "short_direction_long_norm": trace_predictor_from_encoded(model._model, replace_norms(short_bert, long_bert), style),
            }
            for condition, trace in conditions.items():
                rates.append(RateRow(voice_path.stem, passage_id, length, condition, rate(trace, spoken, True), rate(trace, spoken, False)))
            for stage in STAGES:
                short_stage = shared(actual.stages[stage], length)
                long_stage = shared(full_trace.stages[stage], length)
                values = compare(short_stage, long_stage)
                divergences.append(DivergenceRow(voice_path.stem, passage_id, length, stage, *values))
    benchmark_path = (
        ROOT
        / "test_speed/results/evidence/reproduced/duration_training/no_diffusion/metrics/requests.csv"
    )
    benchmark = pd.read_csv(benchmark_path)
    for _, row in benchmark.groupby("voice_id", sort=True).head(1).iterrows():
        voice_path = BENCHMARK_DIR / "embeddings" / f"{row['voice_id']}.pth"
        style = torch.load(voice_path, map_location="cuda", weights_only=True)["voice_vector"].reshape(1, 256)
        phonemes, _ = phonemizer.process_text_with_original_spans(str(row["text"]))
        tokens = torch.tensor([[0, *phonemizer.tokenize(phonemes)]], device="cuda")
        symbols = [phonemizer.index_to_symbol[int(token)] for token in tokens[0]]
        spoken = torch.tensor([bool(symbol.strip()) for symbol in symbols], device="cuda")
        traced = rate(trace_duration_path(model._model, tokens, tokens.shape[1], style), spoken, False)
        expected = float(row["phonemes_per_second"])
        crosschecks.append(CrosscheckRow(str(row["voice_id"]), str(row["text_id"]), expected, traced, abs(expected - traced)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rates": [asdict(row) for row in rates],
            "divergences": [asdict(row) for row in divergences],
            "crosschecks": [asdict(row) for row in crosschecks],
        },
        OUTPUT_DIR / "raw.pt",
    )
    print(f"Raw investigation data: {OUTPUT_DIR / 'raw.pt'}")


if __name__ == "__main__":
    main()
