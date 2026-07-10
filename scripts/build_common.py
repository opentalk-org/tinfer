"""Shared offline helpers for building the 3 weight-input TensorRT engines and
dumping engine plans, weights, and reference IO for the C++/CUDA runtime.

This module is used only at BUILD time (python/torch). The runtime is C++."""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tinfer"))

# AdaIN normalization exports as a native InstanceNormalization node (TRT
# INormalizationLayer) instead of a slow manual mean/var chain: ~40% faster decoder.
os.environ.setdefault("TINFER_ADAIN_NATIVE", "1")

from tinfer.models.impl.styletts2.model.model import StyleTTS2  # noqa: E402
from tinfer.models.impl.styletts2.model.trt3.core import (  # noqa: E402
    promote_initializers_to_inputs, extract_weights, build_engine, WeightInputRunner, dump_tensors)

TEXT_150 = ("Jest to szczególnie przydatne w programowaniu, gdy chcemy szybko generować "
            "mowę o wysokiej jakości dla wielu różnych tekstów w naprawdę krótkim czasie.")


def load_model(model_path: str) -> StyleTTS2:
    m = StyleTTS2(device="cuda")
    m.load(model_path, device="cuda", compile_model=False, load_style_encoder=False, runtime_engine="none")
    m._build_sampler()
    return m


def strip_weight_norm(module: nn.Module) -> None:
    for mod in module.modules():
        try:
            nn.utils.remove_weight_norm(mod)
        except Exception:
            pass


def export_promote_build(
    wrap: nn.Module,
    example_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    profile_shapes: dict,
    out_dir: Path,
    tag: str,
    *,
    trt_export_flag: bool = True,
    promote: bool = True,
    strongly_typed: bool = True,
) -> tuple[Path, dict, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{tag}.onnx"
    wi_path = out_dir / f"{tag}_wi.onnx"
    engine_path = out_dir / f"{tag}.engine"
    if trt_export_flag:
        os.environ["TINFER_TRT_EXPORT"] = "1"
    try:
        with torch.no_grad():
            torch.onnx.export(wrap, example_inputs, str(onnx_path), input_names=input_names,
                              output_names=output_names, opset_version=20, dynamo=False,
                              do_constant_folding=not os.environ.get("TINFER_NO_CONSTFOLD"),
                              dynamic_axes=dynamic_axes)
    finally:
        os.environ.pop("TINFER_TRT_EXPORT", None)
    mp = onnx.load(str(onnx_path))
    weights = extract_weights(mp)
    wnames = promote_initializers_to_inputs(mp) if promote else []
    onnx.save(mp, str(wi_path))
    prof = dict(profile_shapes)
    for n in wnames:
        w = weights[n]
        prof[n] = (tuple(w.shape), tuple(w.shape), tuple(w.shape))
    build_engine(wi_path, engine_path, prof, strongly_typed=strongly_typed)
    return engine_path, {n: weights[n] for n in wnames}, wnames
