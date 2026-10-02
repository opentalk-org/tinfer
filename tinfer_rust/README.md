# Rust runtime

- `src/engine/`: stream coordination, scheduling, batching, and result delivery.
- `src/models/`: model implementations and native execution backends.
- `src/server/`: gRPC and HTTP/WebSocket interfaces.
- `proto/`: canonical gRPC service definition used by the server and client examples.
- `tests/`: integration and backend tests.
- `pysbd/`: sentence segmentation crate.
- `espeak_align/`: phonemization and alignment crate.
- `config.yaml`: CPU configuration using `artifacts/models/magda_rust`.

Run from the repository root:

```bash
nix develop --command cargo run --manifest-path tinfer_rust/Cargo.toml --features onnx -- tinfer_rust/config.yaml
nix develop --command cargo test --manifest-path tinfer_rust/Cargo.toml --features onnx
```

Model export tools live in `tools/model_conversion/`; Python clients live in
`examples/python/`.
