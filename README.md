# Tinfer Rust

Streaming StyleTTS2 inference with gRPC and ElevenLabs-compatible HTTP/WebSocket APIs.
The Rust engine schedules streams, batches model calls, and cuts text into synthesis units.
StyleTTS2 generates audio through native ONNX Runtime or TensorRT execution.

## Run on CPU

```bash
nix develop
cargo run --manifest-path tinfer_rust/Cargo.toml --features onnx -- tinfer_rust/config.yaml
```

Set the model's `backend` to `onnx`, `device` to `cpu`, and `path` to its export directory.
The export must contain `model.toml`, `voices/*.tinf`, and `onnx/cpu/{A,BC}.{onnx,tinf}`.
The configured model settings in `tinfer_rust/config.yaml` are required.

```bash
curl -H 'Content-Type: application/json' \
  -d '{"text":"Dzień dobry.","model_id":"magda"}' \
  'http://localhost:8000/v1/text-to-speech/magda_001?output_format=pcm_24000' \
  -o speech.pcm
```

## Model conversion

Python is used only for model export. The model definitions under `tinfer/` support
these tools; there is no Python inference engine or server.

```bash
uv sync
uv run python tools/styletts2_model_scripts/convert_model.py model_sources/magda \
  -o converted_models/magda_rust --backend onnx --onnx-device cpu \
  --symbols-file tools/styletts2_model_scripts/styletts2_polish_symbols.json \
  --supported-languages pl --default-language pl
uv run python tools/styletts2_model_scripts/convert_voices.py model_sources/magda \
  model_sources/magda/voices/magda_001.wav -o converted_models/magda_rust/voices
```

Use `--onnx-device cpu` for CPU export, `cuda` for GPU export, or `both` for both variants.
CUDA export requires a CUDA device.
TensorRT conversion additionally requires `uv sync --extra tensorrt`.

## Validation

```bash
nix develop --command cargo test --manifest-path tinfer_rust/Cargo.toml --features onnx
uv run pytest tools/styletts2_model_scripts/tests
```

`nix build .#tinfer-rust` builds the CPU server; `nix build .#tinfer-server` builds its OCI image.
Mount your exported model and supply a matching YAML configuration when running the image.

Python HTTP, WebSocket, and gRPC clients are retained in [examples](examples/README.md).
