# StyleTTS2 conversion

This package exports model bundles and voice vectors for the Rust server.
Run `uv sync` from the repository root, then use `uv run tinfer-convert-model`
or `uv run tinfer-convert-voices`. Both commands provide `--help`.

- `src/styletts2_conversion/`: command entry points and ONNX/TensorRT export code.
- `src/styletts2_conversion/modules/`: model definitions required to load checkpoints.
- `src/styletts2_conversion/voice/`: reference voice encoding.
- `symbols/`: explicit language vocabularies for model export.
- `tests/`: converter and export contract tests.

See the repository README for a complete CPU export command.
