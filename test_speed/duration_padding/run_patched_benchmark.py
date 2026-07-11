from dataclasses import replace
from pathlib import Path

import test_speed.run_benchmark as benchmark_runner
from test_speed.benchmark_config import MAGDA_TARGET
from test_speed.benchmark_inference import load_model
from test_speed.duration_padding.benchmark_patch import install_packed_lstm_forward


ROOT = Path(__file__).resolve().parents[2]
PATCHED_TARGET = replace(
    MAGDA_TARGET,
    name="magda_packed_lstm",
    results_dir=ROOT / "test_speed/results_magda_packed_lstm",
    no_diffusion_results_dir=(
        ROOT / "test_speed/results_magda_packed_lstm_no_diffusion"
    ),
)


def load_patched_model(model_path: Path, device: str, runtime_engine: str):
    model = load_model(model_path, device, runtime_engine)
    install_packed_lstm_forward(model._model)
    return model


def main() -> None:
    benchmark_runner.configure_progress_output()
    benchmark_runner.disable_speed_correction()
    benchmark_runner.load_model = load_patched_model
    benchmark_runner._run_target(PATCHED_TARGET)


if __name__ == "__main__":
    main()
