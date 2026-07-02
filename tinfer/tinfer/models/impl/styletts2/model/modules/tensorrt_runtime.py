"""StyleTTS2-specific TensorRT runtime policy: engine naming and shape buckets.

Engine loading, binding and execution are generic and live in trtc.runtime;
this module keeps only what is about *this* model plus thin accessors with the
historical API used by model.py and the process worker.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from trtc.runtime import EngineRunner, clear_runner_cache, get_engine_runner


def next_power_of_two(value: int, minimum: int = 1, maximum: int | None = None) -> int:
    value = max(int(value), int(minimum))
    result = 1 << (value - 1).bit_length()
    if maximum is not None:
        result = min(result, int(maximum))
    return result


@dataclass(frozen=True)
class DiffusionShapeBucket:
    batch_size: int
    embedding_tokens: int
    num_steps: int


@dataclass(frozen=True)
class DynamicDecoderTRTEngineSpec:
    @property
    def file_name(self) -> str:
        return "decoder_dynamic.engine"

    @property
    def onnx_name(self) -> str:
        return "decoder_dynamic.onnx"


@dataclass(frozen=True)
class DynamicDiffusionTRTEngineSpec:
    num_steps: int

    @classmethod
    def from_file_name(cls, file_name: str) -> "DynamicDiffusionTRTEngineSpec":
        match = re.fullmatch(r"diffusion_dynamic_s(\d+)\.engine", Path(file_name).name)
        if match is None:
            raise ValueError(f"Invalid dynamic TensorRT diffusion engine filename: {file_name}")
        return cls(num_steps=int(match.group(1)))

    @property
    def file_name(self) -> str:
        return f"diffusion_dynamic_s{self.num_steps}.engine"

    @property
    def onnx_name(self) -> str:
        return f"diffusion_dynamic_s{self.num_steps}.onnx"


def _share_tensorrt_runners() -> bool:
    return os.getenv("TINFER_TRT_SHARE_RUNNERS", "1") != "0"


def clear_tensorrt_runner_cache() -> None:
    clear_runner_cache()


def get_tensorrt_decoder_runner(engine_dir: str | Path) -> EngineRunner:
    return get_engine_runner(
        Path(engine_dir) / DynamicDecoderTRTEngineSpec().file_name,
        name="decoder",
        share=_share_tensorrt_runners(),
    )


def get_tensorrt_diffusion_runner(engine_dir: str | Path, *, num_steps: int) -> EngineRunner:
    return get_engine_runner(
        Path(engine_dir) / DynamicDiffusionTRTEngineSpec(num_steps=num_steps).file_name,
        name=f"diffusion_s{num_steps}",
        share=_share_tensorrt_runners(),
    )
