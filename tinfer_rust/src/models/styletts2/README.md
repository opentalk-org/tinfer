# StyleTTS2 export contract

One model export has this layout:

```text
model.toml
voices/<voice>.tinf
onnx/cpu/{A,BC}.onnx
onnx/cpu/{A,BC}.tinf
onnx/cuda/{A,BC}.onnx
onnx/cuda/{A,BC}.tinf
tensorrt/{A,BC}.engine
tensorrt/{A,BC}.tinf
```

Only the configured backend/device directory is opened. `architecture_id` identifies
compatible graph shapes, allowing graph programs to be shared across model entries.
Weights, voice files, execution contexts, and streaming state belong to each model entry.
Voice files are loaded on first use.

`model.toml` defines `architecture_id`, `sample_rate = 24000`, `default_language`,
`supported_languages`, and an ordered one-character `symbols` array with `$` at index zero.

A predicts durations, text features, and style. BC generates audio from fixed windows:
32 left-context frames, 128 core frames, and 16 right-context frames. Each frame is
25 ms; only the core audio is emitted. BC takes `en`, `asr`, `s`, `ref`, `phase`, and
`source_noise`, and returns `audio` and `next_phase`.

CPU ONNX uses float32 activations. CUDA ONNX and TensorRT use float16 activations;
weight bundles retain the graph inputs' declared types. Model weights are graph inputs
with matching names in the TINF bundles. Batch and A token dimensions are dynamic.

TINF is little-endian: `TINF`, an i32 tensor count, then each tensor's i32 UTF-8 name
length and name, i32 dtype (`0=f16`, `1=f32`, `2=i32`, `3=i64`, `4=bool`), i32 rank,
i64 dimensions, and tightly packed tensor bytes. A voice file contains 256 float values.

Use `uv run tinfer-convert-model --backend onnx --onnx-device cpu`
for a CPU export. CUDA exports and TensorRT compilation require a CUDA device.
`convert_voices.py` exports voice embeddings from WAV files.
