"""trtc bundle for StyleTTS2: everything this model compiles to TensorRT.

Entry point for `trtc compile <this file> <converted model dir or model.pth>`.
The model-specific residue lives here and in tensorrt_export.py (adapters);
export/build/serve mechanics are trtc's.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from trtc import Axis, Bundle, Component, T


def resolve_converted_model_path(path: str | Path) -> Path:
    converted_path = Path(path)
    if converted_path.is_dir():
        converted_path = converted_path / "model.pth"
    if not converted_path.exists():
        raise FileNotFoundError(f"Converted model not found: {converted_path}")
    if converted_path.name != "model.pth":
        raise ValueError(f"Expected a converted model.pth file, got: {converted_path}")
    return converted_path


@contextmanager
def _trt_export_mode() -> Iterator[None]:
    """Flip the model's ONNX-friendly code paths (manual iSTFT, manual
    instance norm) on for the duration of tracing."""
    previous = os.environ.get("TINFER_TRT_EXPORT")
    os.environ["TINFER_TRT_EXPORT"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TINFER_TRT_EXPORT", None)
        else:
            os.environ["TINFER_TRT_EXPORT"] = previous


def bundle(
    converted_model: str,
    *,
    components: Sequence[str] | str = ("decoder", "diffusion"),
    dtype: str = "float16",
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 16,
    min_asr_frames: int = 128,
    opt_asr_frames: int = 256,
    max_asr_frames: int = 1024,
    min_tokens: int = 8,
    opt_tokens: int = 256,
    max_tokens: int = 512,
    diffusion_steps: Sequence[int] | int = (10,),
    workspace_gb: float = 4.0,
    device: str = "cuda",
) -> Bundle:
    import torch

    from tinfer.models.impl.styletts2.model.model import StyleTTS2

    from .tensorrt_export import (
        DecoderTRTExportModule,
        DiffusionTRTExportModule,
        remove_decoder_weight_norm,
    )
    from .tensorrt_runtime import DynamicDecoderTRTEngineSpec, DynamicDiffusionTRTEngineSpec

    requested = [components] if isinstance(components, str) else list(components)
    steps_list = [diffusion_steps] if isinstance(diffusion_steps, int) else [int(s) for s in diffusion_steps]
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    model_path = resolve_converted_model_path(converted_model)
    model_dir = model_path.parent
    model = StyleTTS2(device=device)
    # Force the pytorch engine: never load engines a prior compile stamped in —
    # we are re-exporting the torch modules, and validating old engines here
    # would wedge re-compiles on a box unlike the builder.
    model.load(
        str(model_path),
        device=device,
        compile_model=False,
        load_style_encoder=False,
        runtime_engine="pytorch",
    )

    batch = Axis("batch", min_batch, opt_batch, max_batch)
    bundle_components: list[Component] = []

    if "decoder" in requested:
        asr_frames = Axis("asr_frames", min_asr_frames, opt_asr_frames, max_asr_frames)
        f0_frames = asr_frames.affine(scale=2, name="f0_frames")
        har_frames = asr_frames.affine(scale=120, offset=1, name="har_frames")

        def decoder_module() -> torch.nn.Module:
            decoder = model._model.decoder.eval().to(device)
            remove_decoder_weight_norm(decoder)
            if torch_dtype == torch.float16:
                decoder = decoder.half()
            return DecoderTRTExportModule(decoder).eval()

        def f0_example(*, shape, device, dtype):
            return torch.rand(shape, device=device, dtype=dtype) * 120.0 + 80.0

        bundle_components.append(
            Component(
                name="decoder",
                module=decoder_module,
                inputs={
                    "asr": T([batch, 512, asr_frames]),
                    "f0": T([batch, f0_frames], example=f0_example),
                    "noise": T([batch, f0_frames]),
                    "style": T([batch, 128]),
                    "har": T([batch, 22, har_frames]),
                },
                outputs=["audio"],
                output_axes={"audio": {0: "batch", 2: "audio_samples"}},
                file_stem=Path(DynamicDecoderTRTEngineSpec().file_name).stem,
                dtype=dtype,
                workspace_gb=workspace_gb,
                export_context=_trt_export_mode,
            )
        )

    if "diffusion" in requested:
        tokens = Axis("embedding_tokens", min_tokens, opt_tokens, max_tokens)

        for num_steps in steps_list:

            def diffusion_module(num_steps: int = num_steps) -> torch.nn.Module:
                diffusion = model._model.diffusion.diffusion.eval().to(device)
                if torch_dtype == torch.float16:
                    diffusion = diffusion.half()
                return DiffusionTRTExportModule(diffusion, num_steps=num_steps).eval()

            bundle_components.append(
                Component(
                    name=f"diffusion_s{num_steps}",
                    module=diffusion_module,
                    inputs={
                        "noise": T([batch, 1, 256]),
                        "step_noise": T([batch, num_steps - 1, 1, 256]),
                        "embedding": T([batch, tokens, 768]),
                        "features": T([batch, 256]),
                    },
                    outputs=["style"],
                    output_axes={"style": {0: "batch"}},
                    file_stem=Path(DynamicDiffusionTRTEngineSpec(num_steps=num_steps).file_name).stem,
                    dtype=dtype,
                    workspace_gb=workspace_gb,
                    export_context=_trt_export_mode,
                    meta={"num_steps": num_steps},
                )
            )

    def finalize(manifest: dict, engine_dir: Path) -> None:
        _write_model_tensorrt_metadata(
            model_path,
            engine_dir=engine_dir,
            components=requested,
            dtype=dtype,
            batch=batch,
            asr_frames=(min_asr_frames, opt_asr_frames, max_asr_frames),
            tokens=(min_tokens, opt_tokens, max_tokens),
            diffusion_steps=steps_list,
            workspace_gb=workspace_gb,
        )
        print(f"updated TensorRT metadata in {model_path}")

    return Bundle(
        name="styletts2",
        components=bundle_components,
        engine_dir_hint=str(model_dir / "tensorrt"),
        finalize=finalize,
    )


def _write_model_tensorrt_metadata(
    model_path: Path,
    *,
    engine_dir: Path,
    components: Sequence[str],
    dtype: str,
    batch: "Axis",
    asr_frames: tuple[int, int, int],
    tokens: tuple[int, int, int],
    diffusion_steps: Sequence[int],
    workspace_gb: float,
) -> None:
    """Stamp runtime_config into model.pth so the existing loader finds the engines."""
    import torch

    from .tensorrt_runtime import DynamicDecoderTRTEngineSpec, DynamicDiffusionTRTEngineSpec

    model_dir = model_path.parent
    try:
        engine_dir_value = str(Path(engine_dir).relative_to(model_dir))
    except ValueError:
        engine_dir_value = str(engine_dir)

    metadata: dict = {
        "engine_dir": engine_dir_value,
        "components": list(components),
        "dtype": dtype,
        "workspace_gb": float(workspace_gb),
    }
    if "decoder" in components:
        metadata["decoder"] = {
            "engine": DynamicDecoderTRTEngineSpec().file_name,
            "min_batch": batch.min,
            "opt_batch": batch.opt,
            "max_batch": batch.max,
            "min_asr_frames": asr_frames[0],
            "opt_asr_frames": asr_frames[1],
            "max_asr_frames": asr_frames[2],
        }
    if "diffusion" in components:
        metadata["diffusion"] = {
            "min_batch": batch.min,
            "opt_batch": batch.opt,
            "max_batch": batch.max,
            "min_tokens": tokens[0],
            "opt_tokens": tokens[1],
            "max_tokens": tokens[2],
            "engines": [
                {"steps": int(steps), "engine": DynamicDiffusionTRTEngineSpec(num_steps=int(steps)).file_name}
                for steps in diffusion_steps
            ],
        }

    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    runtime_config = dict(saved.get("runtime_config") or {})
    runtime_config["engine"] = "tensorrt"
    runtime_config["tensorrt"] = metadata
    saved["runtime_config"] = runtime_config
    torch.save(saved, model_path)
