# StyleTTS2 export contract

One configured model path has this layout:

```text
model.toml
voices/<voice>.tinf
onnx/cpu/{A,B,C}.onnx
onnx/cpu/{A,B,C,glue}.tinf
onnx/cuda/{A,B,C}.onnx
onnx/cuda/{A,B,C,glue}.tinf
tensorrt/{A,B,C}.engine
tensorrt/{A,B,C,glue}.tinf
```

Only `onnx/cpu`, `onnx/cuda`, or `tensorrt` selected by the configured backend and
device is opened. `architecture_id` in `model.toml`
identifies compatible graph shapes, so graph programs are shared by model entries
with the same architecture, backend, and CUDA device. Weight bundles and voice
files remain specific to the configured model path. Voice files are read on first
use; dropping or unloading the last model entry releases its execution contexts,
weights, streams, and graph programs.

`model.toml` contains `architecture_id`, `sample_rate = 24000`,
`default_language`, `supported_languages`, and the ordered one-character `symbols`
array. `$` is symbol zero.

The three graphs use named, dynamic tensors. Model weights are ordinary graph
inputs and have the same names as their TINF entries.

- A request inputs: `tokens`, `mask`, `ref_s`, `noise`, `step_noise`, `alpha`,
  `beta`, `scale`, `sigmas`, `use_diffusion`, `previous_s`, `has_previous`, and
  `style_interpolation`. A outputs: `dur`, `d`, `t_en`, `s`, and `ref`.
- B runtime inputs: `en` and `s`. B outputs: `f0` and `noise`.
- C runtime inputs: `asr`, `f0`, `noise`, `style`, and `har`. C outputs `audio`.
- `glue.tinf` contains `linW` with 9 elements and scalar `linB`.

CPU ONNX graph activations and weights are `f32`. CUDA ONNX and TensorRT graph
activations are `f16`; their weight bundles retain each graph input's required
dtype. Integer and boolean request tensors retain their declared types. Batch,
token, predicted-frame, and audio dimensions are dynamic.

TINF is little-endian: `TINF`, an `i32` tensor count, then for each tensor an `i32`
UTF-8 name length and name, `i32` dtype (`0=f16`, `1=f32`, `2=i32`, `3=i64`,
`4=bool`), `i32` rank, `i64` dimensions, and tightly packed tensor bytes.
