from dataclasses import replace
from pathlib import Path

from test_speed.benchmark.benchmark_config import MAGDA_TARGET
from test_speed.benchmark.run_benchmark import (
    _run_target,
    configure_progress_output,
    disable_speed_correction,
)


ROOT = Path(__file__).resolve().parents[3]
FINETUNED_TARGET = replace(
    MAGDA_TARGET,
    name="magda_duration_finetune",
    model_path=Path(
        "/workspace/converted_models/experiments/magda_duration_consistency/model.pth"
    ),
    results_dir=ROOT / "test_speed/results/evidence/reproduced/duration_training/diffusion",
    no_diffusion_results_dir=(
        ROOT / "test_speed/results/evidence/reproduced/duration_training/no_diffusion"
    ),
    runtime_engine="tensorrt",
)


def main() -> None:
    configure_progress_output()
    disable_speed_correction()
    _run_target(FINETUNED_TARGET)


if __name__ == "__main__":
    main()
