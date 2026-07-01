from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinfer.models.impl.styletts2.model.model import StyleTTS2
from tinfer.models.impl.styletts2.model.modules.tensorrt_export import (
    build_decoder_engine_from_onnx,
    export_decoder_onnx,
)
from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import DecoderTRTEngineSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build StyleTTS2 decoder TensorRT engines.")
    parser.add_argument("--model", default="/workspace/converted_models/magda/model.pth")
    parser.add_argument("--out-dir", default="tmp_tests/trt_engines")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--frames", type=int, nargs="+", default=[256])
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model = StyleTTS2(device="cuda")
    model.load(args.model, device="cuda", compile_model=False)
    decoder = model._model.decoder

    for batch_size in args.batches:
        for asr_frames in args.frames:
            spec = DecoderTRTEngineSpec(batch_size=batch_size, asr_frames=asr_frames)
            onnx_path = out_dir / spec.onnx_name
            engine_path = out_dir / spec.file_name

            if engine_path.exists() and not args.force:
                print(f"skip existing {engine_path}")
                continue

            print(f"export {onnx_path}")
            export_decoder_onnx(
                decoder,
                out_dir,
                batch_size=batch_size,
                asr_frames=asr_frames,
                dtype=dtype,
                device="cuda",
            )

            print(f"build {engine_path}")
            build_decoder_engine_from_onnx(onnx_path, engine_path)
            print(f"built {engine_path} bytes={engine_path.stat().st_size}")


if __name__ == "__main__":
    main()
