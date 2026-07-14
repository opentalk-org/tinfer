from __future__ import annotations

import argparse
from enum import Enum
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tinfer.models.impl.styletts2.model.modules.load_utils import load_original_styletts2_model
from tools.styletts2_model_scripts.artifacts import architecture_id, stage_output, write_manifest


class Backend(str, Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    ALL = "all"


def find_model_files(model_folder: Path) -> tuple[Path, Path]:
    checkpoints = sorted(model_folder.glob("*.pth"))
    configs = sorted((*model_folder.glob("*.yml"), *model_folder.glob("*.yaml")))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one .pth checkpoint in {model_folder}, found {len(checkpoints)}")
    if len(configs) != 1:
        raise ValueError(f"expected exactly one YAML config in {model_folder}, found {len(configs)}")
    return checkpoints[0], configs[0]


def load_symbols(path: Path, expected_count: int) -> tuple[str, ...]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or any(not isinstance(value, str) or len(value) != 1 for value in values):
        raise ValueError("symbols file must contain one-character strings")
    symbols = tuple(values)
    if len(symbols) != expected_count:
        raise ValueError(f"symbol count {len(symbols)} does not match model token count {expected_count}")
    if symbols[0] != "$":
        raise ValueError("symbols must start with $")
    return symbols


def resolve_config_paths(config: dict[str, object], model_folder: Path) -> dict[str, object]:
    resolved = dict(config)
    for key in ("ASR_config", "ASR_path", "F0_path", "PLBERT_dir"):
        value = resolved[key] if key in resolved else None
        if isinstance(value, str) and value.startswith("Utils/"):
            path = (model_folder / value).resolve()
            if not path.exists():
                raise FileNotFoundError(f"config path does not exist: {path}")
            resolved[key] = str(path)
    return resolved


def convert_model(
    model_folder: Path,
    output: Path,
    backend: Backend,
    symbols_file: Path,
    supported_languages: tuple[str, ...],
    default_language: str,
    max_batch: int,
    max_tokens: int,
    max_frames: int,
    workspace_gb: int,
    force: bool,
) -> None:
    if not model_folder.is_dir():
        raise NotADirectoryError(f"original model folder does not exist: {model_folder}")
    if len(set(supported_languages)) != len(supported_languages) or default_language not in supported_languages:
        raise ValueError("supported languages must be unique and contain the default language")
    if max_batch < 1 or max_tokens < 8 or max_frames < 128 or workspace_gb < 1:
        raise ValueError("export requires max_batch >= 1, max_tokens >= 8, max_frames >= 128, and workspace_gb >= 1")

    checkpoint, config_path = find_model_files(model_folder)
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError("StyleTTS2 YAML config must contain a mapping")
    resolved = resolve_config_paths(source_config, model_folder)
    with TemporaryDirectory(prefix="tinfer-styletts2-config-") as temporary:
        resolved_path = Path(temporary) / "config.yml"
        resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        model, model_config = load_original_styletts2_model(str(checkpoint), str(resolved_path))

    symbols = load_symbols(symbols_file, model_config.n_token)
    parameters = []
    for component_name, component in model.items():
        parameters.extend(
            (f"{component_name}.{name}", tuple(parameter.shape)) for name, parameter in component.named_parameters()
        )
    architecture = architecture_id(model_config, parameters, max_batch, max_tokens, max_frames, 5)
    write_manifest(output, architecture, default_language, supported_languages, symbols)
    (output / "voices").mkdir(exist_ok=True)

    if backend in (Backend.ONNX, Backend.ALL):
        # Backend modules are loaded only when selected so ONNX conversion has no TensorRT dependency.
        compiler = importlib.import_module("tools.styletts2_model_scripts.onnx_export")
        with stage_output(output / "onnx", force) as staging:
            compiler.export_onnx(model, model_config, staging, max_tokens, max_frames, 5)
    if backend in (Backend.TENSORRT, Backend.ALL):
        compiler = importlib.import_module("tools.styletts2_model_scripts.tensorrt_export")
        with stage_output(output / "tensorrt", force) as staging:
            compiler.export_tensorrt(model, model_config, staging, max_batch, max_tokens, max_frames, 5, workspace_gb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an original StyleTTS2 model for tinfer_rust")
    parser.add_argument("model_folder", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--backend", choices=[backend.value for backend in Backend], required=True)
    parser.add_argument("--symbols-file", type=Path, required=True)
    parser.add_argument("--supported-languages", nargs="+", required=True)
    parser.add_argument("--default-language", required=True)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    convert_model(
        arguments.model_folder,
        arguments.output,
        Backend(arguments.backend),
        arguments.symbols_file,
        tuple(arguments.supported_languages),
        arguments.default_language,
        arguments.max_batch,
        arguments.max_tokens,
        arguments.max_frames,
        arguments.workspace_gb,
        arguments.force,
    )


if __name__ == "__main__":
    main()
