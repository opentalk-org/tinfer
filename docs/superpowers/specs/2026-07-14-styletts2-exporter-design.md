# StyleTTS2 Exporter Design

## Goal

Replace the fragmented StyleTTS2 conversion scripts with one model exporter and
one independent voice exporter. The produced directory must be directly loadable
by `tinfer_rust`, with no Python conversion step at runtime.

The exporter reads an original StyleTTS2 model directory containing its checkpoint
and YAML configuration. It exports graph programs and external model weights for
CPU ONNX, CUDA ONNX, TensorRT, or all requested targets.

## Commands

`convert_model.py` is the only model conversion entrypoint:

```text
convert_model.py ORIGINAL_MODEL_DIR --output OUTPUT \
  --backend onnx|tensorrt|all \
  --symbols-file SYMBOLS --default-language LANGUAGE \
  --supported-language LANGUAGE [--supported-language LANGUAGE ...] \
  --max-batch N --max-tokens N --max-frames N \
  --max-diffusion-steps N [--workspace-gb N] [--force]
```

`--backend onnx` exports both CPU/f32 and CUDA/mixed-f16 ONNX targets.
`--backend tensorrt` exports only TensorRT. `--backend all` exports all three.
TensorRT compilation imports TensorRT only in the TensorRT module; ONNX export
must not import or initialize TensorRT.

`convert_voices.py` remains separate:

```text
convert_voices.py ORIGINAL_MODEL_DIR WAV [WAV ...] --output OUTPUT/voices
```

It loads the original model's style encoders, converts stereo audio to mono,
extracts one 256-element `f32` `ref_s` tensor per WAV, and writes `<stem>.tinf`.
Empty, non-finite, duplicate-stem, or invalid outputs fail explicitly.

## Output Contract

```text
OUTPUT/
  model.toml
  voices/
  onnx/
    cpu/A.onnx
    cpu/B.onnx
    cpu/C.onnx
    cpu/A.tinf
    cpu/B.tinf
    cpu/C.tinf
    cpu/glue.tinf
    cuda/A.onnx
    cuda/B.onnx
    cuda/C.onnx
    cuda/A.tinf
    cuda/B.tinf
    cuda/C.tinf
    cuda/glue.tinf
  tensorrt/
    A.engine
    B.engine
    C.engine
    A.tinf
    B.tinf
    C.tinf
    glue.tinf
```

Each target owns its matching weights because CPU and GPU programs use different
precisions. A TINF bundle may contain mixed dtypes; CUDA export does not blindly
cast parameters that must remain `f32`.

The Rust loader selects exactly one target directory from backend and device. It
opens no other program or weight directory. CPU is valid only with ONNX. CUDA ONNX
uses `onnx/cuda`; TensorRT uses `tensorrt`.

## Manifest and Program Sharing

`model.toml` contains:

- `architecture_id`
- `sample_rate = 24000`
- `default_language`
- ordered `supported_languages`
- ordered one-character `symbols`, with `$` at index zero

`architecture_id` is a stable hash of the export ABI, topology/configuration,
parameter names and shapes, diffusion-step limit, and graph/profile limits. It
allows the native runtime to share one immutable graph program among models with
the same architecture on the same backend and device while keeping weights and
voices model-specific.

An exporter invocation builds each requested target in a sibling staging
directory and renames it only after all graphs and weights for that target are
valid. Existing output requires `--force`. Adding a backend to an existing output
is allowed only when its manifest is identical; incompatible manifests fail.

## Graph Export

`model_graphs.py` contains the real A/B/C wrappers derived from the current native
export scripts. It owns model-specific graph partitioning but no backend compiler.
The original checkpoint is loaded directly; no intermediate converted `model.pth`
is written.

Graph A accepts request style controls in addition to tokens, masks, reference
style, and diffusion noise. It always produces a diffusion candidate, selects the
reference style when diffusion is disabled, optionally interpolates with previous
style, then applies alpha/beta blending. Per-request guidance scale is reshaped for
correct batch broadcasting. Dynamic sequence handling uses an exportable LSTM
operation rather than a Python loop captured at the example length.

Graph B predicts F0 and noise from aligned text features and style. Graph C runs
the decoder from ASR features, F0, noise, style, and harmonic features. Batch,
token, predicted-frame, and audio dimensions remain dynamic. Constant folding is
disabled so weight-derived values do not become model-specific graph constants.

`onnx_export.py` exports the CPU/f32 and CUDA/mixed-f16 variants and extracts model
parameters into the adjacent TINF files. `tensorrt_export.py` alone creates the
temporary promoted ONNX representation and TensorRT engines with explicit dynamic
profiles. Temporary compiler files do not become part of the output contract.

`artifacts.py` owns TINF serialization, manifest serialization, architecture-ID
construction, and target staging. It has no TensorRT dependency.

## Diffusion Schedule

For a request with `K` diffusion steps, Rust provides `K` sigma values from maximum
to minimum using denominator `K - 1`. The final minimum value is repeated through
the graph's maximum step count, making unused transitions identities. This keeps a
single bounded graph compatible with lower requested step counts.

## Cleanup

The obsolete fragmented model and voice conversion entrypoints under
`tools/styletts2_model_scripts` are removed after their behavior is incorporated.
Backend-neutral code stays outside the ONNX and TensorRT compiler modules. The
symbol JSON files remain usable as explicit command inputs.

## Verification

Focused tests cover TINF byte layout and round-trip parsing, deterministic and
incompatible architecture IDs, manifest compatibility, target selection/staging,
voice tensor validation, and the Rust diffusion schedule. Import tests prove that
ONNX export does not import TensorRT. Rust tests verify backend/device path
selection. A real model fixture is not present in the repository, so full A/B/C
compilation is verified through syntax/import checks and reported as requiring an
external original model directory.
