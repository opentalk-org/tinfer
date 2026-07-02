from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any


def get_tensorrt_model_config(
    runtime_config: Mapping[str, Any],
    model_dir: str | Path,
) -> dict[str, Any] | None:
    if runtime_config.get("engine") != "tensorrt":
        return None

    tensorrt_config = runtime_config.get("tensorrt")
    if not isinstance(tensorrt_config, Mapping):
        raise RuntimeError("engine is set to tensorrt, but the model does not contain TensorRT metadata")

    resolved = dict(tensorrt_config)
    engine_dir = Path(str(resolved.get("engine_dir", "tensorrt")))
    if not engine_dir.is_absolute():
        engine_dir = Path(model_dir) / engine_dir
    resolved["engine_dir"] = str(engine_dir)
    return resolved
