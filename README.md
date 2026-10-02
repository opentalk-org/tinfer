# Tinfer Rust

Streaming StyleTTS2 inference with gRPC and ElevenLabs-compatible HTTP/WebSocket APIs.
The Rust engine schedules streams, batches model calls, and cuts text into synthesis units.
StyleTTS2 generates audio through native ONNX Runtime or TensorRT execution.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `tinfer_rust/` | Rust server, engine, native backends, and Rust tests |
| `tools/model_conversion/` | Python model/voice conversion package and its tests |
| `examples/` | Python, browser, and Electron clients |
| `docs/astro/` | Documentation website |
| `docs/development/history/` | Archived implementation reports |
| `nix/` | Development environment and server/image packaging |
| `artifacts/` | Ignored source models, converted bundles, and generated audio |

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

Python is used only for model export. The model definitions under `tools/model_conversion/` support
these tools; there is no Python inference engine or server.

```bash
uv sync
uv run tinfer-convert-model artifacts/sources/magda \
  -o artifacts/models/magda_rust --backend onnx --onnx-device cpu \
  --symbols-file tools/model_conversion/symbols/styletts2_polish_symbols.json \
  --supported-languages pl --default-language pl
uv run tinfer-convert-voices artifacts/sources/magda \
  artifacts/sources/magda/voices/magda_001.wav -o artifacts/models/magda_rust/voices
```

Use `--onnx-device cpu` for CPU export, `cuda` for GPU export, or `both` for both variants.
CUDA export requires a CUDA device.
TensorRT conversion additionally requires `uv sync --package tinfer-model-conversion --extra tensorrt`.

## Validation

```bash
nix develop --command cargo test --manifest-path tinfer_rust/Cargo.toml --features onnx
uv run pytest tools/model_conversion/tests
```

`nix build .#tinfer-rust` builds the CPU server; `nix build .#tinfer-server` builds its OCI image.
Mount your exported model and supply a matching YAML configuration when running the image.

Python HTTP, WebSocket, and gRPC clients are retained in [examples](examples/README.md).
