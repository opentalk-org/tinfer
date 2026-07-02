from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def use_tensorrt(env: Mapping[str, str]) -> bool:
    value = env.get("USE_TENSORT", env.get("USE_TENSORRT", ""))
    return value.lower() in TRUE_VALUES


def get_tensorrt_model_config(
    runtime_config: Mapping[str, Any],
    env: Mapping[str, str],
    model_dir: str | Path,
) -> dict[str, Any] | None:
    if not use_tensorrt(env):
        return None

    tensorrt_config = runtime_config.get("tensorrt")
    if not isinstance(tensorrt_config, Mapping):
        raise RuntimeError("USE_TENSORT is enabled, but the model does not contain TensorRT metadata")

    resolved = dict(tensorrt_config)
    engine_dir = Path(str(resolved.get("engine_dir", "tensorrt")))
    if not engine_dir.is_absolute():
        engine_dir = Path(model_dir) / engine_dir
    resolved["engine_dir"] = str(engine_dir)
    return resolved
