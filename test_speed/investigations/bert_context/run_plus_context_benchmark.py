from dataclasses import replace
from pathlib import Path

import torch

import test_speed.benchmark.run_benchmark as benchmark_runner
from test_speed.benchmark.benchmark_config import MAGDA_TARGET
from test_speed.benchmark.benchmark_corpus import POLISH_PASSAGE, build_phoneme_grid
from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.bert_context.plus_context import install_plus_context


ROOT = Path(__file__).resolve().parents[3]
EXTRA_TOKENS = 100
MAX_INPUT_TOKENS = 411
POINT_COUNT = 40
TARGET = replace(
    MAGDA_TARGET,
    name="magda_duration_finetune_bert_plus_100",
    model_path=Path(
        "/workspace/converted_models/experiments/magda_duration_consistency/model.pth"
    ),
    results_dir=ROOT / "test_speed/results/evidence/reproduced/bert_plus_context/diffusion",
    no_diffusion_results_dir=(
        ROOT
        / "test_speed/results/evidence/reproduced/bert_plus_context/no_diffusion"
    ),
)


def load_context_model(model_path: Path, device: str, runtime_engine: str):
    model = load_model(model_path, device, runtime_engine)
    phonemizer = model._get_phonemizer("pl")
    phonemes, _ = phonemizer.process_text_with_original_spans(POLISH_PASSAGE)
    passage_tokens = torch.tensor(phonemizer.tokenize(phonemes))
    install_plus_context(model._model.bert, passage_tokens, EXTRA_TOKENS)
    return model


def build_context_grid(
    model,
    passage: str,
    point_count: int,
    max_tokens: int,
    language: str,
    use_training_phonemes: bool,
):
    assert max_tokens == 511, "Unexpected standard benchmark token limit"
    return build_phoneme_grid(
        model,
        passage,
        POINT_COUNT,
        MAX_INPUT_TOKENS,
        language,
        use_training_phonemes,
    )


def main() -> None:
    benchmark_runner.configure_progress_output()
    benchmark_runner.disable_speed_correction()
    benchmark_runner.load_model = load_context_model
    benchmark_runner.build_phoneme_grid = build_context_grid
    benchmark_runner._run_target(TARGET)


if __name__ == "__main__":
    main()
