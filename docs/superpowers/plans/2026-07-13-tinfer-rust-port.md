# Tinfer Rust Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python serving runtime with one Rust process, one unified scheduler, in-process model-owned C++ pipelines, Rust network services, and an exactly compatible Python frontend.

**Architecture:** One Rust library crate is shared by the server executable and PyO3 extension. Core consumes explicit placements; the higher engine emits them automatically. One scheduler actor dispatches immutable jobs into bounded CPU/device executors, while one retained CUDA primary context per GPU is shared across every model replica and execution slot.

**Tech Stack:** Rust 2024, Tokio, Tonic, Axum, Tower, Serde, cxx, CMake, C++20, ONNX Runtime, TensorRT, CUDA Driver/Runtime, PyO3, maturin, NumPy, aiohttp compatibility transport.

## Global Constraints

- Treat `/workspace/tinfer/tinfer/tinfer/` in the current working tree as the behavioral reference.
- Keep every source and plan file under 300 lines and every folder under 16 direct files.
- Keep model implementation inside `tinfer_rust/src/models/<model_name>/`; native model code belongs in its `cpp/` subtree.
- Compile all model `cpp/` trees into one in-process native library; create no C++ executable, inference subprocess, or model sidecar.
- Retain one CUDA primary context per active GPU and share it across replicas, streams, ONNX Runtime, TensorRT, and model-native CUDA stages.
- Core placement is explicit; only the higher engine may choose devices automatically.
- Use one logical scheduler actor and one mutable scheduling-state owner.
- Copy `tinfer/espeak_align/` as a separate Cargo project; do not rewrite or remove `tools/styletts2_model_scripts/`.
- Use deterministic stub artifacts; model export, PyTorch architecture, training, and checkpoint conversion are outside this port.
- Preserve the current gRPC, HTTP, WebSocket, audio, alignment, and Python package contracts.
- Add no backend, device, or error fallback; invalid configuration fails explicitly.

---

## Ordered Stage Plans

1. [Foundation, core, and unified scheduler](2026-07-13-tinfer-rust-stage-1-foundation-scheduler.md)
2. [espeak, unified native build, stub model, CUDA, and TensorRT](2026-07-13-tinfer-rust-stage-2-native-model.md)
3. [Audio and gRPC services](2026-07-13-tinfer-rust-stage-3-audio-grpc.md)
4. [HTTP and ElevenLabs-compatible WebSockets](2026-07-13-tinfer-rust-stage-4-http-websocket.md)
5. [Python compatibility, operations, and cutover](2026-07-13-tinfer-rust-stage-5-python-cutover.md)

Stages are sequential. Within a stage, execute tasks in order because each task's interfaces are consumed by the next. Every task ends in a narrow commit; every stage ends with an independently runnable binary or test harness.

## Repository-Wide Verification Gate

Run from `/workspace/tinfer/tinfer_rust` after every stage:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cmake --build target/cmake-build --target tinfer_models_native
```

Expected: all commands exit 0. Hardware-tagged CUDA and TensorRT tests may be selected only on workers providing the declared libraries and devices; they must never replace a missing backend with CPU.

Before cutover, run these separately from `/workspace/tinfer`:

```bash
python -m pytest tmp_tests/tts_api tmp_tests/styletts2 -q
python -m pytest tinfer_rust/python_tests -q
python tinfer_rust/tools/check_api_manifest.py --reference tinfer --candidate tinfer-rust-wheel
```

Expected: all suites pass and the manifest checker reports `0 incompatible API differences`.
