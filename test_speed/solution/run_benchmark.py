from dataclasses import replace
from pathlib import Path

import test_speed.benchmark.run_benchmark as benchmark_runner
from test_speed.benchmark.benchmark_config import MAGDA_TARGET
from test_speed.solution.balanced_corpus import build_balanced_grid


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path("/workspace/converted_models/magda_aligned_rate/model.pth")
RESULTS_ROOT = ROOT / "test_speed/results/final/magda_rate"
NESTED_TARGET = replace(
    MAGDA_TARGET,
    name="magda_aligned_rate",
    model_path=MODEL_PATH,
    results_dir=RESULTS_ROOT / "nested",
    no_diffusion_results_dir=RESULTS_ROOT / "nested_no_diffusion",
)
INDEPENDENT_TARGET = replace(
    NESTED_TARGET,
    name="magda_aligned_rate_independent",
    results_dir=RESULTS_ROOT / "independent",
    no_diffusion_results_dir=RESULTS_ROOT / "independent_no_diffusion",
)


def main() -> None:
    benchmark_runner.configure_progress_output()
    benchmark_runner.disable_speed_correction()
    benchmark_runner._run_target(NESTED_TARGET)
    benchmark_runner.build_phoneme_grid = build_balanced_grid
    benchmark_runner._run_target(INDEPENDENT_TARGET)


if __name__ == "__main__":
    main()
