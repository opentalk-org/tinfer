from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tinfer"))


def resolve_converted_model_path(path: str | Path) -> Path:
    converted_path = Path(path)
    if converted_path.is_dir():
        converted_path = converted_path / "model.pth"
    if not converted_path.exists():
        raise FileNotFoundError(f"Converted model not found: {converted_path}")
    if converted_path.name != "model.pth":
        raise ValueError(f"Expected a converted model.pth file, got: {converted_path}")
    return converted_path


def _relative_to_model_dir(path: Path, model_dir: Path) -> str:
    try:
        return str(path.relative_to(model_dir))
    except ValueError:
        return str(path)


def build_dynamic_tensorrt_metadata(
    *,
    engine_dir: Path,
    model_dir: Path,
    components: Sequence[str],
    dtype: str,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    min_asr_frames: int,
    opt_asr_frames: int,
    max_asr_frames: int,
    min_tokens: int,
    opt_tokens: int,
    max_tokens: int,
    diffusion_steps: Sequence[int],
    workspace_gb: float,
) -> dict:
    metadata = {
        "engine_dir": _relative_to_model_dir(engine_dir, model_dir),
        "components": list(components),
        "dtype": dtype,
        "workspace_gb": float(workspace_gb),
    }
    if "decoder" in components:
        metadata["decoder"] = {
            "engine": "decoder_dynamic.engine",
            "min_batch": int(min_batch),
            "opt_batch": int(opt_batch),
            "max_batch": int(max_batch),
            "min_asr_frames": int(min_asr_frames),
            "opt_asr_frames": int(opt_asr_frames),
            "max_asr_frames": int(max_asr_frames),
        }
    if "diffusion" in components:
        metadata["diffusion"] = {
            "min_batch": int(min_batch),
            "opt_batch": int(opt_batch),
            "max_batch": int(max_batch),
            "min_tokens": int(min_tokens),
            "opt_tokens": int(opt_tokens),
            "max_tokens": int(max_tokens),
            "engines": [
                {"steps": int(steps), "engine": f"diffusion_dynamic_s{int(steps)}.engine"}
                for steps in diffusion_steps
            ],
        }
    return metadata


def update_model_tensorrt_metadata(model_path: str | Path, metadata: dict) -> None:
    model_path = Path(model_path)
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    runtime_config = dict(saved.get("runtime_config") or {})
    runtime_config["tensorrt"] = metadata
    saved["runtime_config"] = runtime_config
    torch.save(saved, model_path)


def _load_compile_dependencies():
    model_module = importlib.import_module("tinfer.models.impl.styletts2.model.model")
    export_module = importlib.import_module("tinfer.models.impl.styletts2.model.modules.tensorrt_export")
    runtime_module = importlib.import_module("tinfer.models.impl.styletts2.model.modules.tensorrt_runtime")
    return model_module.StyleTTS2, export_module, runtime_module


def compile_converted_model(
    converted_model: str | Path,
    *,
    components: Sequence[str],
    engine_dir: str | Path | None = None,
    dtype: str = "float16",
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 16,
    min_asr_frames: int = 128,
    opt_asr_frames: int = 256,
    max_asr_frames: int = 512,
    min_tokens: int = 16,
    opt_tokens: int = 128,
    max_tokens: int = 256,
    diffusion_steps: Sequence[int] = (10,),
    workspace_gb: float = 4.0,
    force: bool = False,
) -> dict:
    StyleTTS2, tensorrt_export, tensorrt_runtime = _load_compile_dependencies()
    model_path = resolve_converted_model_path(converted_model)
    model_dir = model_path.parent
    engine_dir = Path(engine_dir) if engine_dir is not None else model_dir / "tensorrt"
    engine_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    workspace_bytes = int(float(workspace_gb) * (1 << 30))

    model = StyleTTS2(device="cuda")
    model.load(str(model_path), device="cuda", compile_model=False, load_style_encoder=False)

    if "decoder" in components:
        spec = tensorrt_runtime.DynamicDecoderTRTEngineSpec()
        onnx_path = engine_dir / spec.onnx_name
        engine_path = engine_dir / spec.file_name
        if force or not engine_path.exists():
            print(f"export {onnx_path}")
            tensorrt_export.export_decoder_dynamic_onnx(
                model._model.decoder,
                engine_dir,
                opt_batch_size=opt_batch,
                opt_asr_frames=opt_asr_frames,
                dtype=torch_dtype,
                device="cuda",
            )
            print(f"build {engine_path}")
            tensorrt_export.build_decoder_dynamic_engine_from_onnx(
                onnx_path,
                engine_path,
                min_batch=min_batch,
                opt_batch=opt_batch,
                max_batch=max_batch,
                min_asr_frames=min_asr_frames,
                opt_asr_frames=opt_asr_frames,
                max_asr_frames=max_asr_frames,
                workspace_bytes=workspace_bytes,
            )
        else:
            print(f"skip existing {engine_path}")

    if "diffusion" in components:
        for steps in diffusion_steps:
            spec = tensorrt_runtime.DynamicDiffusionTRTEngineSpec(num_steps=int(steps))
            onnx_path = engine_dir / spec.onnx_name
            engine_path = engine_dir / spec.file_name
            if force or not engine_path.exists():
                print(f"export {onnx_path}")
                tensorrt_export.export_diffusion_dynamic_onnx(
                    model._model.diffusion.diffusion,
                    engine_dir,
                    opt_batch_size=opt_batch,
                    opt_embedding_tokens=opt_tokens,
                    num_steps=int(steps),
                    dtype=torch_dtype,
                    device="cuda",
                )
                print(f"build {engine_path}")
                tensorrt_export.build_diffusion_dynamic_engine_from_onnx(
                    onnx_path,
                    engine_path,
                    min_batch=min_batch,
                    opt_batch=opt_batch,
                    max_batch=max_batch,
                    min_tokens=min_tokens,
                    opt_tokens=opt_tokens,
                    max_tokens=max_tokens,
                    num_steps=int(steps),
                    workspace_bytes=workspace_bytes,
                )
            else:
                print(f"skip existing {engine_path}")

    metadata = build_dynamic_tensorrt_metadata(
        engine_dir=engine_dir,
        model_dir=model_dir,
        components=components,
        dtype=dtype,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        min_asr_frames=min_asr_frames,
        opt_asr_frames=opt_asr_frames,
        max_asr_frames=max_asr_frames,
        min_tokens=min_tokens,
        opt_tokens=opt_tokens,
        max_tokens=max_tokens,
        diffusion_steps=diffusion_steps,
        workspace_gb=workspace_gb,
    )
    update_model_tensorrt_metadata(model_path, metadata)
    print(f"updated TensorRT metadata in {model_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile dynamic TensorRT engines for a converted StyleTTS2 model.")
    parser.add_argument("converted_model", help="Path to converted model directory or converted model.pth")
    parser.add_argument("--engine-dir", default=None, help="Engine output directory (default: <model_dir>/tensorrt)")
    parser.add_argument("--components", nargs="+", choices=["decoder", "diffusion"], default=["decoder", "diffusion"])
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--min-asr-frames", type=int, default=128)
    parser.add_argument("--opt-asr-frames", type=int, default=256)
    parser.add_argument("--max-asr-frames", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--opt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--diffusion-steps", type=int, nargs="+", default=[10])
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compile_converted_model(
        args.converted_model,
        components=args.components,
        engine_dir=args.engine_dir,
        dtype=args.dtype,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        min_asr_frames=args.min_asr_frames,
        opt_asr_frames=args.opt_asr_frames,
        max_asr_frames=args.max_asr_frames,
        min_tokens=args.min_tokens,
        opt_tokens=args.opt_tokens,
        max_tokens=args.max_tokens,
        diffusion_steps=args.diffusion_steps,
        workspace_gb=args.workspace_gb,
        force=args.force,
    )


if __name__ == "__main__":
    main()
