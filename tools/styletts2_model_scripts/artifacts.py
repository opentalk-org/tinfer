from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
from typing import Any
from uuid import uuid4

import numpy as np


DTYPE_CODES = {
    np.dtype("float16"): 0,
    np.dtype("float32"): 1,
    np.dtype("int32"): 2,
    np.dtype("int64"): 3,
    np.dtype("bool"): 4,
}


def write_tinf(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    if len(tensors) != len(set(tensors)) or any(not name for name in tensors):
        raise ValueError("TINF tensor names must be non-empty and unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(b"TINF")
        output.write(struct.pack("<i", len(tensors)))
        for name, source in tensors.items():
            value = np.ascontiguousarray(source)
            dtype = value.dtype.newbyteorder("=")
            if dtype not in DTYPE_CODES:
                raise TypeError(f"unsupported TINF dtype: {value.dtype}")
            encoded_name = name.encode("utf-8")
            output.write(struct.pack("<i", len(encoded_name)))
            output.write(encoded_name)
            output.write(struct.pack("<ii", DTYPE_CODES[dtype], value.ndim))
            output.write(struct.pack(f"<{value.ndim}q", *value.shape))
            output.write(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())


def architecture_id(
    model_config: Any,
    parameters: Iterable[tuple[str, tuple[int, ...]]],
    max_batch: int,
    max_tokens: int,
    max_frames: int,
    max_diffusion_steps: int,
) -> str:
    description = {
        "abi": "styletts2-weight-input-v1",
        "config": _canonical(model_config),
        "parameters": sorted((name, list(shape)) for name, shape in parameters),
        "limits": [max_batch, max_tokens, max_frames, max_diffusion_steps],
    }
    encoded = json.dumps(description, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return f"styletts2-{sha256(encoded).hexdigest()[:16]}"


def write_manifest(
    root: Path,
    architecture: str,
    default_language: str,
    supported_languages: Iterable[str],
    symbols: Iterable[str],
) -> None:
    content = "\n".join(
        (
            f"architecture_id = {json.dumps(architecture)}",
            "sample_rate = 24000",
            f"default_language = {json.dumps(default_language, ensure_ascii=False)}",
            f"supported_languages = {_toml_array(supported_languages)}",
            f"symbols = {_toml_array(symbols)}",
            "",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "model.toml"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError("output contains an incompatible StyleTTS2 manifest")
        return
    temporary = root / f".model.toml.{uuid4().hex}"
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


@contextmanager
def stage_output(destination: Path, force: bool) -> Iterator[Path]:
    if destination.exists() and not force:
        raise FileExistsError(f"target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        yield staging
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot hash architecture value of type {type(value).__name__}")


def _toml_array(values: Iterable[str]) -> str:
    return "[" + ", ".join(json.dumps(value, ensure_ascii=False) for value in values) + "]"
