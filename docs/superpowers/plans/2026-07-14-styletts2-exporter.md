# StyleTTS2 Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export original StyleTTS2 checkpoints into backend-specific artifacts that `tinfer_rust` loads directly, plus export WAV voice embeddings separately.

**Architecture:** One typed artifact module defines TINF, manifests, targets, and atomic staging. Model-specific A/B/C PyTorch wrappers are shared by two isolated compilers: ONNX exports CPU and CUDA graphs, while TensorRT alone imports TensorRT and builds engines from temporary ONNX. Rust/C++ select one exact backend/device directory and never load unused weights.

**Tech Stack:** Python 3.12, PyTorch, ONNX, TensorRT, NumPy, soundfile, pytest, Rust 2024, C++20.

## Global Constraints

- Work only in `/workspace/tinfer-worktrees/tinfer-rust-port`; do not commit on main.
- Do not use subagents.
- Keep every file below 300 lines and every folder below 16 entries.
- Imports remain at module tops; TensorRT is imported only by `tensorrt_export.py`.
- The original checkpoint and YAML are direct inputs; no converted `model.pth` is produced.
- `--backend onnx` means CPU and CUDA ONNX; `tensorrt` means TensorRT only; `all` means all targets.
- CPU weights/activations are `f32`; GPU bundles may have mixed required dtypes.
- Tests exercise behavior, not source-file existence.

---

### Task 1: Typed artifacts and target staging

**Files:**
- Create: `tools/styletts2_model_scripts/artifacts.py`
- Create: `tools/styletts2_model_scripts/tests/test_artifacts.py`

**Interfaces:**
- Produces: `Backend`, `Target`, `TensorValue`, `Manifest`, `targets_for()`, `architecture_id()`, `write_tinf()`, `write_manifest()`, `staged_target()`.
- Consumes: NumPy arrays and `pathlib.Path` only; no torch, ONNX, or TensorRT imports.

- [ ] **Step 1: Write failing artifact tests**

Test exact little-endian TINF header/name/dtype/shape/data bytes; verify architecture hashes are deterministic and change with topology/profile inputs; verify backend expansion; verify staging replaces a target only after success and rejects incompatible manifests.

- [ ] **Step 2: Verify RED**

Run: `/workspace/tinfer/.venv/bin/python -m pytest tools/styletts2_model_scripts/tests/test_artifacts.py -q`

Expected: collection fails because `artifacts` does not exist.

- [ ] **Step 3: Implement artifacts**

Use frozen dataclasses and enums:

```python
class Backend(Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    ALL = "all"

@dataclass(frozen=True)
class Target:
    backend: Backend
    device: str
    relative_dir: Path

@dataclass(frozen=True)
class Manifest:
    architecture_id: str
    sample_rate: int
    default_language: str
    supported_languages: tuple[str, ...]
    symbols: tuple[str, ...]
```

`write_tinf(path, tensors)` serializes contiguous arrays with dtype codes
`float16=0`, `float32=1`, `int32=2`, `int64=3`, `bool=4`. `architecture_id()`
hashes canonical JSON containing ABI, model config, sorted parameter shapes, and
limits. `staged_target()` yields a sibling temporary directory and atomically
renames it on clean exit.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused pytest command, then commit only Task 1 files with
`feat(export): add StyleTTS2 artifact contract`.

### Task 2: Model graph partitions and ONNX exporter

**Files:**
- Create: `tools/styletts2_model_scripts/model_graphs.py`
- Create: `tools/styletts2_model_scripts/onnx_export.py`
- Create: `tools/styletts2_model_scripts/tests/test_onnx_export.py`

**Interfaces:**
- Consumes: original model object, `Target`, output directory, export limits.
- Produces: `EngineA`, `EngineB`, `EngineC`, `export_onnx_target(model, target, output, limits)`.

- [ ] **Step 1: Write failing graph/helper tests**

Test style selection/interpolation/blending with a tiny tensor-only helper; test
guidance scale reshapes from `[B]` to `[B,1,1]`; import `onnx_export` and assert
`tensorrt` is absent from `sys.modules`; test initializer promotion/extraction on a
small ONNX model.

- [ ] **Step 2: Verify RED**

Run the focused pytest file and confirm missing modules/functions cause failure.

- [ ] **Step 3: Port the real graph partitions**

Move the working A/B/C definitions from `scripts/build_a.py`, `build_b.py`, and
`build_c.py` without their hard-coded capture paths. Engine A uses native dynamic
LSTM states, a bounded ADPM2 loop, per-request `use_diffusion`, previous-style
interpolation, and alpha/beta blending. Engine B wraps predictor F0/noise. Engine C
wraps decoder/generator and keeps required virtual split outputs for weight-input
engine stability.

- [ ] **Step 4: Implement ONNX-only compilation**

Export opset 20 with `dynamo=False`, `do_constant_folding=False`, named inputs and
outputs, and dynamic batch/token/frame/audio axes. Export `onnx/cpu` in f32 and
`onnx/cuda` with the model-specific mixed precision. Promote initializers to graph
inputs, save their arrays into adjacent `A.tinf`, `B.tinf`, and `C.tinf`, and write
`glue.tinf` from `l_linear` weight/bias.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused tests and `python -m py_compile` for both modules; commit with
`feat(export): add StyleTTS2 ONNX graphs`.

### Task 3: TensorRT compiler and conversion command

**Files:**
- Create: `tools/styletts2_model_scripts/tensorrt_export.py`
- Create: `tools/styletts2_model_scripts/convert_model.py`
- Create: `tools/styletts2_model_scripts/tests/test_convert_model.py`
- Delete: `tools/styletts2_model_scripts/compile_converted_model.py`
- Delete: `tools/styletts2_model_scripts/convert_model_folder.py`
- Delete: `tools/styletts2_model_scripts/convert_json_to_pth.py`
- Delete: `tools/styletts2_model_scripts/text_config_cli.py`

**Interfaces:**
- Produces: `ExportLimits`, `load_original_model()`, `convert_model()`, CLI `main()`; `export_tensorrt_target()`.
- Consumes: Task 1 targets/artifacts and Task 2 graph wrappers.

- [ ] **Step 1: Write failing command tests**

Test strict discovery of exactly one checkpoint and one YAML; explicit symbol and
language validation; backend dispatch order; existing compatible manifest behavior;
and rejection of `--workspace-gb` when TensorRT is not selected.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm failure is from missing conversion module.

- [ ] **Step 3: Implement TensorRT-only compilation**

Only this module imports TensorRT. It consumes the shared graph wrappers, creates
temporary promoted ONNX per stage, configures strongly typed explicit-batch
profiles from `max_batch`, `max_tokens`, and `max_frames`, builds `A.engine`,
`B.engine`, and `C.engine`, serializes matching weight bundles, then removes the
temporary ONNX files.

- [ ] **Step 4: Implement the dispatcher**

Use a frozen `ExportLimits` dataclass. Discover exactly one `*.pth` and one
`*.yml|*.yaml`, resolve `Utils/` paths against the supplied original folder, load
with `load_original_styletts2_model`, validate symbols against `model_config.n_token`,
compute one architecture ID, write/validate `model.toml`, and invoke selected
targets through `staged_target()`. No backend compiler is imported until its target
is selected.

- [ ] **Step 5: Verify GREEN and commit**

Run command tests, artifact/ONNX tests, and `py_compile`; commit with
`feat(export): add StyleTTS2 conversion command`.

### Task 4: Separate WAV voice exporter

**Files:**
- Create: `tools/styletts2_model_scripts/convert_voices.py`
- Create: `tools/styletts2_model_scripts/tests/test_convert_voices.py`
- Delete: `tools/styletts2_model_scripts/convert_wavs_to_voices.py`

**Interfaces:**
- Produces: `validate_voice_vector()`, `convert_voices()`, CLI `main()`.
- Consumes: original model directory, one or more WAV paths, output voice directory, Task 1 `write_tinf()`.

- [ ] **Step 1: Write failing voice tests**

Test mono and stereo waveform normalization, exact `[256]` finite f32 vector
validation, duplicate WAV stems, and that written TINF contains one tensor named
`ref_s` with shape `[256]`.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm missing voice module causes failure.

- [ ] **Step 3: Implement voice conversion**

Load the original model with style encoders, choose CUDA when available otherwise
CPU, read each explicit WAV through soundfile, average channel axis for stereo,
call `StyleTTS2VoiceEncoder.compute_style_from_waveform`, validate, and write
`<stem>.tinf`. Reject empty inputs and collisions before loading the model.

- [ ] **Step 4: Verify GREEN and commit**

Run voice and artifact tests plus `py_compile`; commit with
`feat(export): add StyleTTS2 voice conversion`.

### Task 5: Runtime target selection and schedule parity

**Files:**
- Modify: `tinfer_rust/src/models/styletts2/cpp/model.cpp`
- Modify: `tinfer_rust/src/models/styletts2/preprocessing.rs`
- Modify: `tinfer_rust/src/models/styletts2/tests.rs`
- Modify: `tinfer_rust/src/models/styletts2/README.md`

**Interfaces:**
- Consumes: Task 1 output layout.
- Produces: exact selected directory loading and a padded maximum-five-step diffusion schedule.

- [ ] **Step 1: Write failing Rust schedule test**

Expose `diffusion_schedule` to the sibling test module and assert that three steps
produce maximum, midpoint, minimum, minimum, minimum, minimum; five steps produce
five distinct maximum-to-minimum values followed by minimum.

- [ ] **Step 2: Verify RED**

Run: `cargo test models::styletts2::tests::diffusion_schedule --manifest-path tinfer_rust/Cargo.toml`

Expected: current `steps + 1` interpolation differs from required schedule.

- [ ] **Step 3: Fix schedule and runtime paths**

Compute `steps` values using denominator `steps - 1`, then pad to six with the
minimum. In C++, choose `onnx/cpu` for device `< 0`, `onnx/cuda` for CUDA ONNX,
and `tensorrt` for TensorRT; load programs and all four TINF bundles from that exact
directory.

- [ ] **Step 4: Update contract documentation**

Replace the old shared `weights/` layout with the approved backend/device layout
and describe mixed GPU weight dtypes accurately.

- [ ] **Step 5: Verify GREEN and commit**

Run focused Rust tests, `cargo fmt --check`, and `cargo clippy --all-targets -- -D warnings`;
commit with `fix(runtime): select StyleTTS2 export target`.

### Task 6: Focused end-to-end verification and main-tree sync

**Files:**
- Modify only files required by failures attributable to Tasks 1–5.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: accepted worktree patch copied to main without a main-branch commit.

- [ ] **Step 1: Run Python verification**

Run all tests under `tools/styletts2_model_scripts/tests`, compile every exporter
module, and show `convert_model.py --help` plus `convert_voices.py --help`.

- [ ] **Step 2: Run Rust verification**

Run StyleTTS2 unit tests, format check, and clippy. Do not run hardware inference
without an original checkpoint fixture.

- [ ] **Step 3: Inspect scope**

Run `git diff --check`, confirm every modified Python/Rust/C++ file is below 300
lines and the script folder is below 16 entries, and inspect the complete diff for
stale hard-coded paths or mixed ONNX/TensorRT imports.

- [ ] **Step 4: Commit worktree verification fixes if any**

Commit only necessary corrections on `codex/tinfer-rust-port`; leave main
uncommitted.

- [ ] **Step 5: Copy the accepted patch to main**

Apply the worktree diff to `/workspace/tinfer`, preserving unrelated main-tree
changes. Verify `git status --short` shows the intended `tinfer_rust` and tools/docs
changes with no main commit.
