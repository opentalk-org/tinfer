from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
import torch.nn as nn

from tools.styletts2_model_scripts.artifacts import write_tinf
from tools.styletts2_model_scripts.model_graphs import EngineA, EngineBC, strip_weight_norm


def promote_weights(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    weights = {initializer.name: numpy_helper.to_array(initializer).copy() for initializer in model.graph.initializer}
    for initializer in model.graph.initializer:
        model.graph.input.append(helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims))
    del model.graph.initializer[:]
    return weights


def export_onnx(
    model: object,
    model_config: object,
    output: Path,
    max_tokens: int,
    max_diffusion_steps: int,
) -> None:
    export_variant(model, model_config, output / "cpu", "cpu", torch.float32, max_tokens, max_diffusion_steps)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ONNX export requires an available CUDA device")
    export_variant(model, model_config, output / "cuda", "cuda", torch.float16, max_tokens, max_diffusion_steps)


def export_variant(
    model: object,
    model_config: object,
    output: Path,
    device: str,
    dtype: torch.dtype,
    max_tokens: int,
    max_diffusion_steps: int,
) -> None:
    output.mkdir(parents=True)
    token_count = min(32, max_tokens)
    frame_count = 176
    wrappers = (EngineA(model, max_diffusion_steps), EngineBC(model.predictor, model.decoder))
    for wrapper in wrappers:
        strip_weight_norm(wrapper)
        wrapper.eval().to(device=device, dtype=dtype)
    if dtype is torch.float16:
        wrappers[0].bert.float()

    old_adain = os.environ.get("TINFER_ADAIN_NATIVE")
    old_export = os.environ.get("TINFER_TRT_EXPORT")
    os.environ["TINFER_ADAIN_NATIVE"] = "1"
    os.environ["TINFER_TRT_EXPORT"] = "1"
    try:
        _export_a(wrappers[0], output, device, dtype, token_count, max_diffusion_steps)
        _export_bc(
            wrappers[1], output, device, dtype, frame_count,
            model.predictor.shared.input_size, model_config.hidden_dim,
        )
    finally:
        _restore_environment("TINFER_ADAIN_NATIVE", old_adain)
        _restore_environment("TINFER_TRT_EXPORT", old_export)

def _export_a(
    wrapper: nn.Module,
    output: Path,
    device: str,
    dtype: torch.dtype,
    tokens: int,
    max_steps: int,
) -> None:
    batch = 1
    inputs = (
        torch.randint(0, 10, (batch, tokens), device=device, dtype=torch.int64),
        torch.zeros(batch, tokens, device=device, dtype=torch.bool),
        torch.zeros(batch, 256, device=device, dtype=dtype),
        torch.randn(batch, 1, 256, device=device, dtype=dtype),
        torch.randn(batch, max_steps - 1, 1, 256, device=device, dtype=dtype),
        torch.full((batch, 1), 0.3, device=device, dtype=dtype),
        torch.full((batch, 1), 0.7, device=device, dtype=dtype),
        torch.ones(batch, device=device, dtype=torch.float32),
        torch.tensor([3.0, 1.0, 0.1, 0.01, 0.0001, 0.0001], device=device, dtype=torch.float32),
        torch.ones(batch, device=device, dtype=torch.bool),
        torch.zeros(batch, 256, device=device, dtype=dtype),
        torch.zeros(batch, device=device, dtype=torch.bool),
        torch.full((batch,), 0.7, device=device, dtype=dtype),
    )
    split_names = [f"dstep{index}" for index in range(max_steps - 1)]
    input_names = [
        "tokens", "mask", "ref_s", "noise", "step_noise", "alpha", "beta", "scale", "sigmas",
        "use_diffusion", "previous_s", "has_previous", "style_interpolation",
    ]
    output_names = ["dur", "d", "t_en", "s", "ref", "bert_dur", "d_en", *split_names]
    dynamic = {
        "tokens": {0: "B", 1: "T"}, "mask": {0: "B", 1: "T"}, "ref_s": {0: "B"}, "noise": {0: "B"},
        "step_noise": {0: "B"}, "alpha": {0: "B"}, "beta": {0: "B"}, "scale": {0: "B"},
        "use_diffusion": {0: "B"}, "previous_s": {0: "B"}, "has_previous": {0: "B"},
        "style_interpolation": {0: "B"}, "dur": {0: "B", 1: "T"}, "d": {0: "B", 1: "T"},
        "t_en": {0: "B", 2: "T"}, "s": {0: "B"}, "ref": {0: "B"}, "bert_dur": {0: "B", 1: "T"},
        "d_en": {0: "B", 2: "T"},
    }
    dynamic.update({name: {0: "B"} for name in split_names})
    _save_weight_input_graph(wrapper, inputs, input_names, output_names, dynamic, output, "A")


def _export_bc(
    wrapper: nn.Module,
    output: Path,
    device: str,
    dtype: torch.dtype,
    frames: int,
    encoding_channels: int,
    asr_channels: int,
) -> None:
    inputs = (
        torch.randn(1, encoding_channels, frames, device=device, dtype=dtype),
        torch.randn(1, asr_channels, frames, device=device, dtype=dtype),
        torch.randn(1, 128, device=device, dtype=dtype),
        torch.randn(1, 128, device=device, dtype=dtype),
        torch.zeros(1, 9, device=device, dtype=torch.float32),
        torch.randn(1, frames * 600, 9, device=device, dtype=dtype),
    )
    dynamic = {
        "en": {0: "B"}, "asr": {0: "B"}, "s": {0: "B"}, "ref": {0: "B"},
        "phase": {0: "B"}, "source_noise": {0: "B"}, "audio": {0: "B"}, "next_phase": {0: "B"},
    }
    _save_weight_input_graph(
        wrapper, inputs, ["en", "asr", "s", "ref", "phase", "source_noise"],
        ["audio", "next_phase"], dynamic, output, "BC",
    )


def _save_weight_input_graph(
    wrapper: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    output: Path,
    stage: str,
) -> None:
    graph_path = output / f"{stage}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, graph_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            opset_version=20, dynamo=False, do_constant_folding=True,
        )
    graph = onnx.load(graph_path)
    weights = promote_weights(graph)
    onnx.checker.check_model(graph)
    onnx.save(graph, graph_path)
    write_tinf(output / f"{stage}.tinf", weights)


def _restore_environment(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
