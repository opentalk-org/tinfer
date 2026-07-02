from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import threading
from typing import Mapping

import tensorrt as trt
import torch


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


ShapeProfile = Mapping[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
_TRT_RUNNER_CACHE: dict[tuple[object, ...], object] = {}
_TRT_RUNNER_CACHE_LOCK = threading.Lock()


def clear_tensorrt_runner_cache() -> None:
    with _TRT_RUNNER_CACHE_LOCK:
        _TRT_RUNNER_CACHE.clear()


def _share_tensorrt_runners() -> bool:
    return os.getenv("TINFER_TRT_SHARE_RUNNERS", "1") != "0"


def _engine_dir_cache_key(engine_dir: str | Path) -> str:
    return str(Path(engine_dir).expanduser().resolve())


def decoder_dynamic_profile_shapes(
    *,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    min_asr_frames: int,
    opt_asr_frames: int,
    max_asr_frames: int,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    def decoder_shapes(batch: int, asr_frames: int) -> dict[str, tuple[int, ...]]:
        f0_frames = asr_frames * 2
        return {
            "asr": (batch, 512, asr_frames),
            "f0": (batch, f0_frames),
            "noise": (batch, f0_frames),
            "style": (batch, 128),
            "har": (batch, 22, asr_frames * 120 + 1),
        }

    min_shapes = decoder_shapes(min_batch, min_asr_frames)
    opt_shapes = decoder_shapes(opt_batch, opt_asr_frames)
    max_shapes = decoder_shapes(max_batch, max_asr_frames)
    return {name: (min_shapes[name], opt_shapes[name], max_shapes[name]) for name in min_shapes}


def diffusion_dynamic_profile_shapes(
    *,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    min_tokens: int,
    opt_tokens: int,
    max_tokens: int,
    num_steps: int,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    def diffusion_shapes(batch: int, tokens: int) -> dict[str, tuple[int, ...]]:
        return {
            "noise": (batch, 1, 256),
            "step_noise": (batch, num_steps - 1, 1, 256),
            "embedding": (batch, tokens, 768),
            "features": (batch, 256),
        }

    min_shapes = diffusion_shapes(min_batch, min_tokens)
    opt_shapes = diffusion_shapes(opt_batch, opt_tokens)
    max_shapes = diffusion_shapes(max_batch, max_tokens)
    return {name: (min_shapes[name], opt_shapes[name], max_shapes[name]) for name in min_shapes}


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


def _torch_dtype_from_trt(dtype) -> torch.dtype:
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT64:
        return torch.int64
    if dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


class _TensorRTRunner:
    def __init__(
        self,
        engine_dir: str | Path,
        spec: DynamicDecoderTRTEngineSpec | DynamicDiffusionTRTEngineSpec,
        component_name: str,
    ) -> None:
        self.engine_dir = Path(engine_dir)
        self.spec = spec
        self.component_name = component_name
        self.engine_path = self.engine_dir / self.spec.file_name
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Missing TensorRT {self.component_name} engine: {self.engine_path}")

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        with trt.Runtime(self._logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")
        self._streams: dict[int, torch.cuda.Stream] = {}
        self._run_lock = threading.Lock()

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self._input_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._output_names)

    def run(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with self._run_lock:
            return self._run_without_lock(inputs)

    def _run_without_lock(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not inputs:
            raise ValueError(f"TensorRT {self.component_name} runner requires input tensors")

        first_input = next(iter(inputs.values()))
        if not first_input.is_cuda:
            raise ValueError(f"TensorRT {self.component_name} inputs must be CUDA tensors")

        bound_inputs: dict[str, torch.Tensor] = {}
        for name in self._input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT {self.component_name} input: {name}")
            tensor = inputs[name].contiguous()
            expected_dtype = _torch_dtype_from_trt(self._engine.get_tensor_dtype(name))
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            self._context.set_input_shape(name, tuple(tensor.shape))
            bound_inputs[name] = tensor

        outputs: dict[str, torch.Tensor] = {}
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                self._context.infer_shapes()
                shape = tuple(self._context.get_tensor_shape(name))
            dtype = _torch_dtype_from_trt(self._engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device=first_input.device, dtype=dtype)

        for name, tensor in bound_inputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())
        for name, tensor in outputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())

        device_index = first_input.device.index if first_input.device.index is not None else torch.cuda.current_device()
        stream = self._streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=first_input.device)
            self._streams[device_index] = stream

        current_stream = torch.cuda.current_stream(first_input.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"TensorRT {self.component_name} execution failed: {self.engine_path}")
        current_stream.wait_stream(stream)
        return outputs


class TensorRTDecoderRunner(_TensorRTRunner):
    def __init__(
        self,
        engine_dir: str | Path,
    ) -> None:
        super().__init__(engine_dir, DynamicDecoderTRTEngineSpec(), "decoder")


class TensorRTDiffusionRunner(_TensorRTRunner):
    def __init__(
        self,
        engine_dir: str | Path,
        *,
        num_steps: int,
    ) -> None:
        super().__init__(engine_dir, DynamicDiffusionTRTEngineSpec(num_steps=num_steps), "diffusion")


def get_tensorrt_decoder_runner(
    engine_dir: str | Path,
) -> TensorRTDecoderRunner:
    if not _share_tensorrt_runners():
        return TensorRTDecoderRunner(engine_dir)

    key = ("decoder", _engine_dir_cache_key(engine_dir))
    with _TRT_RUNNER_CACHE_LOCK:
        runner = _TRT_RUNNER_CACHE.get(key)
        if runner is None:
            runner = TensorRTDecoderRunner(engine_dir)
            _TRT_RUNNER_CACHE[key] = runner
        return runner


def get_tensorrt_diffusion_runner(
    engine_dir: str | Path,
    *,
    num_steps: int,
) -> TensorRTDiffusionRunner:
    if not _share_tensorrt_runners():
        return TensorRTDiffusionRunner(engine_dir, num_steps=num_steps)

    key = (
        "diffusion",
        _engine_dir_cache_key(engine_dir),
        num_steps,
    )
    with _TRT_RUNNER_CACHE_LOCK:
        runner = _TRT_RUNNER_CACHE.get(key)
        if runner is None:
            runner = TensorRTDiffusionRunner(engine_dir, num_steps=num_steps)
            _TRT_RUNNER_CACHE[key] = runner
        return runner
