from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import re

import torch


def next_power_of_two(value: int, minimum: int = 1, maximum: int | None = None) -> int:
    value = max(int(value), int(minimum))
    result = 1 << (value - 1).bit_length()
    if maximum is not None:
        result = min(result, int(maximum))
    return result


@dataclass(frozen=True)
class DecoderShapeBucket:
    batch_size: int
    asr_frames: int
    f0_frames: int
    audio_samples: int


def decoder_shape_bucket(
    batch_size: int,
    asr_frames: int,
    *,
    max_batch: int,
    max_frames: int,
) -> DecoderShapeBucket:
    bucket_batch = next_power_of_two(batch_size, minimum=1, maximum=max_batch)
    bucket_asr = next_power_of_two(asr_frames, minimum=128, maximum=max_frames)
    return DecoderShapeBucket(
        batch_size=bucket_batch,
        asr_frames=bucket_asr,
        f0_frames=bucket_asr * 2,
        audio_samples=bucket_asr * 600,
    )


@dataclass(frozen=True)
class DecoderTRTEngineSpec:
    batch_size: int
    asr_frames: int

    @classmethod
    def from_file_name(cls, file_name: str) -> "DecoderTRTEngineSpec":
        match = re.fullmatch(r"decoder_b(\d+)_t(\d+)\.engine", Path(file_name).name)
        if match is None:
            raise ValueError(f"Invalid TensorRT decoder engine filename: {file_name}")
        return cls(batch_size=int(match.group(1)), asr_frames=int(match.group(2)))

    @property
    def file_name(self) -> str:
        return f"decoder_b{self.batch_size}_t{self.asr_frames}.engine"

    @property
    def onnx_name(self) -> str:
        return f"decoder_b{self.batch_size}_t{self.asr_frames}.onnx"


def _torch_dtype_from_trt(dtype) -> torch.dtype:
    import tensorrt as trt

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


class TensorRTDecoderRunner:
    def __init__(self, engine_dir: str | Path, *, batch_size: int, asr_frames: int) -> None:
        self.engine_dir = Path(engine_dir)
        self.spec = DecoderTRTEngineSpec(batch_size=batch_size, asr_frames=asr_frames)
        self.engine_path = self.engine_dir / self.spec.file_name
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Missing TensorRT decoder engine: {self.engine_path}")

        import tensorrt as trt

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
        if not inputs:
            raise ValueError("TensorRT decoder runner requires input tensors")

        first_input = next(iter(inputs.values()))
        if not first_input.is_cuda:
            raise ValueError("TensorRT decoder inputs must be CUDA tensors")

        bound_inputs: dict[str, torch.Tensor] = {}
        for name in self._input_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT decoder input: {name}")
            tensor = inputs[name].contiguous()
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
                raise RuntimeError(f"TensorRT decoder execution failed: {self.engine_path}")
        current_stream.wait_stream(stream)
        return outputs
