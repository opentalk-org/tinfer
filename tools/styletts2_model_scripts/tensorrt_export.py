from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import tensorrt as trt
import torch

from tools.styletts2_model_scripts.onnx_export import export_variant


def profile_shapes(
    stage: str,
    name: str,
    shape: tuple[int, ...],
    max_batch: int,
    max_tokens: int,
    max_frames: int,
    max_steps: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if all(dimension > 0 for dimension in shape):
        return shape, shape, shape
    batch = (1, 1, max_batch)
    if stage == "A" and name in ("tokens", "mask"):
        return (1, 8), (1, min(32, max_tokens)), (max_batch, max_tokens)
    if stage == "A" and name in ("ref_s", "previous_s"):
        return (1, 256), (1, 256), (max_batch, 256)
    if stage == "A" and name == "noise":
        return (1, 1, 256), (1, 1, 256), (max_batch, 1, 256)
    if stage == "A" and name == "step_noise":
        return (1, max_steps - 1, 1, 256), (1, max_steps - 1, 1, 256), (max_batch, max_steps - 1, 1, 256)
    if stage == "A" and name in ("alpha", "beta"):
        return (1, 1), (1, 1), (max_batch, 1)
    if stage == "A" and name in ("scale", "use_diffusion", "has_previous", "style_interpolation"):
        return (1,), (1,), (max_batch,)
    if stage == "B" and name == "en":
        channels = shape[1]
        return (1, channels, 32), (1, channels, min(128, max_frames)), (max_batch, channels, max_frames)
    if stage == "B" and name == "s":
        return (1, 128), (1, 128), (max_batch, 128)
    if stage == "C" and name == "asr":
        channels = shape[1]
        return (1, channels, 128), (1, channels, min(128, max_frames)), (max_batch, channels, max_frames)
    if stage == "C" and name in ("f0", "noise"):
        return (1, 256), (1, min(128, max_frames) * 2), (max_batch, max_frames * 2)
    if stage == "C" and name == "style":
        return (1, 128), (1, 128), (max_batch, 128)
    if stage == "C" and name == "har":
        return (1, 22, 15361), (1, 22, min(128, max_frames) * 120 + 1), (max_batch, 22, max_frames * 120 + 1)
    raise ValueError(f"no TensorRT profile for dynamic input {stage}.{name} with shape {shape}")


def export_tensorrt(
    model: object,
    model_config: object,
    output: Path,
    max_batch: int,
    max_tokens: int,
    max_frames: int,
    max_steps: int,
    workspace_gb: int,
) -> None:
    with TemporaryDirectory(prefix="tinfer-styletts2-trt-") as temporary:
        graphs = Path(temporary) / "onnx"
        export_variant(model, model_config, graphs, "cuda", torch.float16, max_tokens, max_frames, max_steps)
        for stage in ("A", "B", "C"):
            _build_engine(
                graphs / f"{stage}.onnx",
                output / f"{stage}.engine",
                stage,
                max_batch,
                max_tokens,
                max_frames,
                max_steps,
                workspace_gb,
            )
            shutil.copy2(graphs / f"{stage}.tinf", output / f"{stage}.tinf")
        shutil.copy2(graphs / "glue.tinf", output / "glue.tinf")


def _build_engine(
    graph_path: Path,
    engine_path: Path,
    stage: str,
    max_batch: int,
    max_tokens: int,
    max_frames: int,
    max_steps: int,
    workspace_gb: int,
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(graph_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {graph_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * 1024**3)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dimension) for dimension in tensor.shape)
        minimum, optimum, maximum = profile_shapes(stage, tensor.name, shape, max_batch, max_tokens, max_frames, max_steps)
        profile.set_shape(tensor.name, minimum, optimum, maximum)
    if not profile or config.add_optimization_profile(profile) < 0:
        raise RuntimeError(f"TensorRT rejected the {stage} optimization profile")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"TensorRT failed to build {stage}")
    engine_path.write_bytes(bytes(engine))
