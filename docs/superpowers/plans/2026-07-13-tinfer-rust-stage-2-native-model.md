# Tinfer Rust Stage 2 Native Model Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add copied espeak alignment and one model-owned native stub pipeline that runs ONNX Runtime or TensorRT graphs around CPU or CUDA stages while sharing one primary context per GPU.

**Architecture:** Model-specific Rust, C++, CUDA, tensor contracts, and pipeline glue all live under `src/models/stub/`. One root CMake target compiles every registered model source list into one static library. Generic native code owns only device primary-context lifetime, RAII, status conversion, and the cxx bridge support used by model wrappers.

**Tech Stack:** Cargo workspace, cxx, CMake, C++20, CUDA Driver/Runtime, ONNX Runtime C++ API, TensorRT C++ API, espeak-ng.

## Global Constraints

- Apply every constraint in `2026-07-13-tinfer-rust-port.md`.
- Do not build a C++ `main`, model DLL, helper process, or backend sidecar.
- Backend/platform selection occurs during pipeline construction, never per request.
- CPU CI must run without CUDA or TensorRT installed; configured unavailable features fail explicitly rather than falling back.

---

### Task 1: Copy and Integrate espeak_align

**Files:**
- Copy: `tinfer/espeak_align/` to `tinfer_rust/espeak_align/`
- Modify: `tinfer_rust/Cargo.toml`
- Create: `tinfer_rust/src/models/stub/{mod.rs,preprocessing.rs,text.rs}`
- Test: `tinfer_rust/tests/espeak_parity.rs`
- Test: `tinfer_rust/tests/fixtures/espeak_cases.json`

**Interfaces:**
- Produces: `StubPreprocessor::prepare(&str, &StubParameters) -> Result<PreparedText>` where `PreparedText` retains original UTF-8 spans and aligned phonemes.

- [ ] **Step 1: Copy the Cargo project without rewriting it**

Run: `cp -a tinfer/espeak_align tinfer_rust/espeak_align`

Expected: `tinfer_rust/espeak_align/espeak_align_core/Cargo.toml` and the separate PyO3 crate exist; no source under `tools/styletts2_model_scripts/` changes.

- [ ] **Step 2: Write failing parity tests**

Create fixtures from current alignment tests and StyleTTS2 phonemizer behavior, including punctuation, whitespace, Unicode, batches, and source spans.

```rust
#[test]
fn preprocessing_preserves_reference_spans() {
    for case in fixtures() {
        assert_eq!(StubPreprocessor::new().prepare(&case.text, &case.params).unwrap(), case.expected);
    }
}
```

Run: `cargo test --test espeak_parity`

Expected: compile failure because `StubPreprocessor` is absent.

- [ ] **Step 3: Add the path dependency and model-owned wrapper**

Call `espeak-align-core` directly. Normalize only rules proved by fixtures, carry aligned source byte ranges through vocabulary filtering, and return typed token IDs plus spans. Do not import the copied PyO3 wrapper into the server crate.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --manifest-path espeak_align/Cargo.toml && cargo test --test espeak_parity`

Expected: copied project tests and parity fixtures pass.

```bash
git add tinfer_rust/espeak_align tinfer_rust/Cargo.toml tinfer_rust/src/models/stub tinfer_rust/tests
git commit -m "feat(rust): copy and integrate espeak alignment"
```

### Task 2: Unified Native Target and Safe Bridge

**Files:**
- Create: `tinfer_rust/build.rs`
- Create: `tinfer_rust/native/CMakeLists.txt`
- Create: `tinfer_rust/native/{device_registry.hpp,device_registry.cpp,status.hpp}`
- Create: `tinfer_rust/src/native/{mod.rs,device.rs}`
- Create: `tinfer_rust/src/models/stub/bridge.rs`
- Create: `tinfer_rust/src/models/stub/cpp/CMakeLists.txt`
- Create: `tinfer_rust/src/models/stub/cpp/include/stub_pipeline.hpp`
- Create: `tinfer_rust/src/models/stub/cpp/pipeline/stub_pipeline.cpp`
- Test: `tinfer_rust/tests/native_lifetime.rs`

**Interfaces:**
- Produces: `DeviceRegistry::retain(DeviceId) -> Result<SharedPtr<DeviceRuntime>>` and opaque `ffi::StubPipeline::{create,warmup,infer}` through `cxx::bridge`.

- [ ] **Step 1: Write a failing lifetime test**

```rust
#[test]
fn pipelines_share_one_device_runtime() {
    let registry = DeviceRegistry::for_test();
    let first = registry.retain(DeviceId::new(0));
    let second = registry.retain(DeviceId::new(0));
    assert_eq!(first.identity(), second.identity());
    assert_eq!(registry.primary_retain_calls(DeviceId::new(0)), 1);
}
```

Run: `cargo test --test native_lifetime --features native-test-double`

Expected: compile failure for missing `DeviceRegistry`.

- [ ] **Step 2: Define the cxx boundary**

```rust
#[cxx::bridge(namespace = "tinfer::stub")]
mod ffi {
    struct NativeBatch { values: Vec<i64>, offsets: Vec<u32>, seeds: Vec<u64> }
    struct NativeOutput { audio: Vec<f32>, offsets: Vec<u32>, durations: Vec<u32> }
    unsafe extern "C++" {
        type StubPipeline;
        fn infer(self: Pin<&mut StubPipeline>, slot: u32, batch: &NativeBatch) -> Result<NativeOutput>;
    }
}
```

Keep pointers opaque, flatten tensors, use fixed-width fields, and catch every C++ exception before it reaches Rust. Rust owns request buffers through the call and copies returned output before releasing the slot.

- [ ] **Step 3: Implement one root CMake target**

`build.rs` runs cxxbridge and one CMake configure. Root CMake adds generic native sources and `add_subdirectory` for the compile-time model registry; each model contributes sources to `tinfer_models_native` with `target_sources`. Assert no executable target exists.

Implement the test-double registry first. Under CUDA, use `cuInit`, `cuDevicePrimaryCtxRetain`, a scoped `cuCtxPushCurrent`/`cuCtxPopCurrent` guard, and release only after pipelines are destroyed.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test native_lifetime --features native-test-double && cmake --build target/cmake-build --target tinfer_models_native`

Expected: shared identity/retain-count test passes and exactly one static native target builds.

```bash
git add tinfer_rust/build.rs tinfer_rust/native tinfer_rust/src/native tinfer_rust/src/models/stub
git commit -m "feat(rust): add unified native model library"
```

### Task 3: Artifact Contract and ONNX CPU Stub

**Files:**
- Create: `tinfer_rust/src/models/stub/{config.rs,manifest.rs,model.rs,postprocessing.rs}`
- Create: `tinfer_rust/src/models/stub/cpp/{onnx/session.hpp,onnx/session.cpp,cpu/fast_stage.hpp,cpu/fast_stage.cpp,pipeline/types.hpp}`
- Create: `tinfer_rust/tests/artifacts/stub/manifest.toml`
- Create: `tinfer_rust/tests/artifacts/stub/graphs/{graph_1.onnx,graph_2.onnx}`
- Test: `tinfer_rust/tests/stub_onnx_cpu.rs`

**Interfaces:**
- Produces: `StubFactory: ModelFactory`, `StubReplica: ModelReplica`, validated `StubManifest`, and pipeline order `graph_1 -> CpuFastStage -> graph_2`.

- [ ] **Step 1: Write manifest and golden-output tests**

```rust
#[test]
fn onnx_cpu_pipeline_matches_golden_audio() {
    let model = load_stub(BackendKind::Onnx, PlatformKind::Cpu).unwrap();
    let output = model.infer(slot(0), prepared_batch(["abc", "de"])).unwrap();
    assert_eq!(output.lengths(), [golden_len("abc"), golden_len("de")]);
    assert_close(output.audio(), golden_audio(), 1e-6);
}
```

Also assert missing graphs, wrong dtype/rank, duplicate voices, unsupported pairs, and out-of-bound shapes fail before warmup.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test stub_onnx_cpu --features onnx`

Expected: compile failure for missing `StubFactory`.

- [ ] **Step 3: Implement the deterministic pipeline**

Manifest fields include revision, sample rate, languages, voices, backend/platform pairs, tensor names/dtypes/ranks/bounds, graph paths, and warmup shapes. Graph 1 emits features and durations; CPU expands features by duration; graph 2 emits float mono audio. Construct backend/platform implementations once and store them in the model-owned pipeline.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test stub_onnx_cpu --features onnx`

Expected: golden batches, validation cases, warmup, and repeated inference pass.

```bash
git add tinfer_rust/src/models/stub tinfer_rust/tests/artifacts tinfer_rust/tests/stub_onnx_cpu.rs
git commit -m "feat(rust): run stub pipeline through ONNX CPU"
```

### Task 4: CUDA, ONNX CUDA, and TensorRT Slots

**Files:**
- Create: `tinfer_rust/src/models/stub/cpp/cuda/{fast_stage.cu,fast_stage.hpp}`
- Create: `tinfer_rust/src/models/stub/cpp/tensorrt/{engine.cpp,engine.hpp}`
- Create: `tinfer_rust/src/models/stub/cpp/pipeline/{slot.cpp,slot.hpp}`
- Modify: `tinfer_rust/src/models/stub/cpp/onnx/session.cpp`
- Test: `tinfer_rust/tests/stub_gpu.rs`
- Test: `tinfer_rust/tests/multigpu.rs`

**Interfaces:**
- Produces: fixed execution-slot pools where each slot owns its stream, buffers, graph bindings, completion events, and TensorRT execution contexts.

- [ ] **Step 1: Write hardware-tagged parity and context tests**

```rust
#[gpu_test]
fn all_backends_match_cpu_and_share_primary_context() {
    let outputs = run_all_supported_stub_pairs();
    assert_all_close(outputs, 1e-5);
    assert_eq!(native_metrics().primary_retain_calls(0), 1);
    assert!(native_metrics().max_concurrent_slots(0) > 1);
}
```

Add a two-GPU fake-inventory test proving explicit placement and concurrent dispatch, plus a real two-GPU test when available.

- [ ] **Step 2: Confirm selected tests fail**

Run: `cargo test --test stub_gpu --features onnx-cuda,tensorrt -- --ignored`

Expected: compile failure for missing CUDA/TensorRT stage implementations.

- [ ] **Step 3: Implement slot-owned backend resources**

Bind ONNX CUDA sessions to the slot's user stream and reusable I/O bindings. Give each TensorRT graph one immutable engine and one execution context per slot. Launch the CUDA fast stage on that same stream. Record events between stages, synchronize only at Rust output ownership, and never call context create/reset in model code.

Classify recoverable placement errors separately from fatal CUDA device errors. Recoverable errors rebuild on the same `DeviceRuntime`; fatal errors disable the device without resetting its shared context.

- [ ] **Step 4: Verify the stage and commit**

Run CPU gate: `cargo test --workspace --features onnx`

Run GPU gate: `cargo test --test stub_gpu --test multigpu --features onnx-cuda,tensorrt -- --ignored`

Expected: CPU remains green; available GPU pairs match golden output, overlap slots, and report one retained primary context per GPU.

```bash
git add tinfer_rust/src/models/stub tinfer_rust/tests/stub_gpu.rs tinfer_rust/tests/multigpu.rs
git commit -m "feat(rust): add shared-context CUDA and TensorRT execution"
```
