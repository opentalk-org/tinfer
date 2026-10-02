# Stage 2 Task 4 Implementation Report

## Status

Complete. CUDA, ONNX Runtime CUDA EP, and TensorRT execute the committed stub graphs in-process through the one `tinfer_models_native` static target.

## TDD Evidence

- RED: `cargo test --test stub_gpu --features onnx-cuda,tensorrt -- --ignored` failed because both requested Cargo features were absent.
- GREEN hardware: the final ignored GPU suite runs both real backends on GPU0 and compares every sample, duration, and offset against the CPU ONNX pipeline.
- The deterministic fake two-device test runs without CUDA hardware and the real inventory test explicitly reports the one-device skip.

## Implementation

- Added explicit `onnx-cuda` and `tensorrt` feature composition. CUDA/test-double combinations fail before dependency acquisition or CMake configuration.
- Added pinned/checksummed GPU ORT 1.22.0, TensorRT v11.1 headers, and patchelf 0.18 acquisition under a process lock. TensorRT and CUDA runtime libraries come from explicit caller roots, are version/file validated, and are staged without production references to this host's venv, Nix store, or worktree.
- Every staged ORT provider and TensorRT library has `$ORIGIN`-only RUNPATH. The test executable also has an origin-relative profile runtime RUNPATH. Hermetic `readelf`/`ldd` tests resolve all provider dependencies without inherited loader variables.
- Added one model-owned CUDA kernel for duration expansion and one model-owned TensorRT parser/builder/runtime implementation. TensorRT builds two immutable engines once per pipeline and creates exactly two execution contexts per slot.
- Added fixed CUDA slots. Each slot owns one non-default stream, completion event, reusable graph buffers, pinned final-output buffers, ONNX CUDA sessions/I/O bindings, and TensorRT contexts. An atomic lease rejects same-slot overlap; separate slots have no shared pipeline lock.
- ONNX CUDA configures each slot session with its user stream, reusable device-bound I/O binding, `CUDAExecutionProvider`, and `session.disable_cpu_ep_fallback=1`.
- GPU order is graph 1 enqueue, CUDA expansion enqueue, graph 2 enqueue, final duration/audio D2H, completion event, and one final event wait. Batch/cardinality/shape arithmetic is validated before launches.
- Added typed `NativeRecoverable` and `NativeDeviceFatal` mappings. Invalid placement is recoverable; fatal device state atomically disables new work. Cleanup can still enter the retained primary context without resetting it.
- Added race-safe primary retain/release and active/max slot metrics. Source audit finds one `cuDevicePrimaryCtxRetain`, no context creation/reset/device reset, one CMake library target, and no native executable.
- Extended the stub artifact and factory to accept ONNX/CPU, ONNX/CUDA, and TensorRT/CUDA while retaining one loaded model per config entry and a fixed slot pool inside that model.

## Verification

- `cargo test --workspace`: pass.
- `cargo test --workspace --features onnx`: pass, including 18 real CPU ONNX tests.
- `cargo test --workspace --features native-test-double`: pass, including deterministic two-device concurrent dispatch.
- `cargo test --test stub_gpu --features onnx-cuda,tensorrt -- --ignored --nocapture`: 2 passed; real ONNX CUDA and TensorRT exact parity within `1e-5`, repeated inference, every slot, mixed lifetime, provider identity, engine/context counts, one retain, and overlap.
- Hermetic direct execution with `env -i PATH=/usr/bin:/bin`: GPU parity test passed.
- `cargo test --test multigpu --features native-cuda -- --ignored --nocapture`: pass with `SKIP: CUDA inventory has 1 device; two are required`.
- Strict Clippy passed separately for default, `onnx`, `onnx-cuda`, `tensorrt`, `native-test-double`, and the combined GPU features.
- Rust formatting passed in both Cargo workspaces.
- Explicit CMake `tinfer_models_native` build passed; target inventory contains no executable target.
- `git diff --check`, source file/folder limits, espeak integrity, and `tools/styletts2_model_scripts` integrity passed.

## Host Constraint

Only GPU0 (RTX 5090, compute capability 12.0) exists, so a real two-GPU overlap cannot be demonstrated on this host. The ignored test executes the real path automatically when inventory is at least two; the deterministic fake inventory covers two distinct runtimes and one retain each here.

## Review Remediation

- TensorRT now validates engine version, exact I/O cardinality, tensor names, modes, dtypes, ranks, static/profile bounds, and resolved output sizes before assigning any device address. Real static-output and dynamically oversized-output ONNX fixtures prove rejection before unsafe binding or allocation.
- Native failures cross the CXX boundary as inference, recoverable-device, or fatal-device errors. Backend exceptions retain their original message and use a context health probe; only fatal failures disable the shared runtime. The native test double exercises both paths.
- GPU batching uses one slot-owned aggregate token region and one aggregate expanded/audio region. Every item queues graph one, the CUDA stage, graph two, and D2H copies at checked offsets, followed by exactly one completion-event synchronization for the whole batch.
- The real multi-GPU test builds ONNX/CUDA on device 0 and TensorRT on device 1, warms both, and infers concurrently after a barrier. This host has one CUDA device, so it reports an explicit skip; the deterministic two-device test remains the executable concurrency proof here.
- GPU dependencies come from one typed `TINFER_GPU_LIBRARY_MANIFEST`. Validation covers absolute roots, strict relative paths, byte size, SHA-256, ELF class/endian/machine, SONAME, required exports, pinned versions, and requested compute-capability builder resources. Staging is digest-addressed, sealed after RUNPATH rewriting, and reused only after full seal verification.
- Each TensorRT engine owns its logger, including synchronized error snapshots, so logger lifetime covers parser, builder, engine, and execution contexts without shared static state.
- GPU manifest contracts and all aggregate capacities are validated before constructing native resources. Allocation counters prove malicious maximum bounds allocate no CUDA resources, and all native allocation/copy/cursor arithmetic uses checked size, byte, and addition helpers.
- The combined-feature CPU validation test previously treated coherent ONNX/CUDA placement as unsupported. It now tests the actual CPU-platform/CUDA-device mismatch and passes under CPU-only and combined GPU feature sets.

## Final Review Verification

- `cargo test --workspace`, `--features onnx`, `--features native-test-double`, and `--features onnx-cuda,tensorrt`: pass.
- Combined ignored GPU suite: 5 passed, covering hermetic runtime linkage, pre-allocation rejection, both TensorRT shape failures, backend parity, context sharing, slot overlap, and one final wait.
- Real multi-GPU inventory test: pass with explicit `SKIP: CUDA inventory has 1 device; two are required`.
- Strict Clippy: pass for default, ONNX, native test double, and combined GPU configurations. `cargo fmt --all -- --check` and `git diff --check`: pass.
- `readelf` confirms executable and staged provider RUNPATHs are `$ORIGIN`-relative; `ldd` resolves ONNX Runtime and TensorRT from the digest-addressed Cargo-profile runtime.
- No model C++ executable target exists. Changed production source files remain below 300 lines, changed folders remain below 16 files, and neither `espeak_align` nor `tools/styletts2_model_scripts` changed.

## Important-Finding Remediation

- Both ONNX CUDA graph runs now receive typed `Ort::RunOptions` with `disable_synchronize_execution_providers=1`. The hardware test warms the pipeline, enqueues a deterministic delay on the slot stream, records an event before `Run`, and proves `Run` returns while that event is still not ready. As a negative control, replacing those options with default run options made the assertion fail; restoring them made all six ignored hardware tests pass. The inference path retains one final completion wait and exact CPU/ONNX CUDA/TensorRT parity.
- Device retain/count/context entry and model create/warmup/infer/identity boundaries return structured native status objects containing an exhaustive error enum and message. Every expected C++ boundary catches native, standard, and unknown exceptions. Regression tests pass messages beginning with the former fatal/recoverable prefixes and prove the enum alone controls runtime disable behavior.
- GPU manifests are validated through retained open descriptors. Staging checks path device/inode identity, stream-copies and hashes the descriptor into a private temporary file under the lock, rehashes the descriptor, verifies exact caller size and SHA-256, checks identity again, and only then publishes. Tests reject synchronized path replacement and in-place mutation without publishing a destination. Generation directories and integrity seals use the full caller-manifest digest and exact source inventory, so a stale seal is rejected after even a comment-only manifest change.
- Mutation entry points and hardware proof hooks are excluded from ordinary production builds. The test double and hardware tests use separate Cargo features and CXX bridges; production archives and test binaries contain none of the disable, context-entry, failure-injection, or allocation-counter symbols. Real GPU tests do not call mutation seams, and allocation-counter state was removed.

## Remediation Verification

- Fresh final-state workspace tests pass for default, `native-test-double`, and `onnx-cuda,tensorrt,native-hardware-tests`. The combined ordinary matrix includes 18 CPU ONNX tests; the six hardware tests pass separately, including the asynchronous-run proof and TensorRT bounds cases.
- Strict Clippy with warnings denied passes for default, ONNX, test double, ONNX CUDA, TensorRT, and combined GPU plus hardware-test configurations.
- Explicit `nm` audits confirm production archive seam absence and test-double archive seam presence. CMake/source audits confirm a single `tinfer_models_native` static library and no native executable declarations.
- Rust formatting in both Cargo workspaces, `git diff --check`, protected-directory integrity, source/folder limits, message-protocol removal, and allocation-counter removal all pass.

## Final Three-Finding Remediation

- The hardware-only ONNX proof now returns structured `asynchronous_event_pending` and `default_event_ready` fields. It runs the same warmed session, slot, stream, graph, token input, delay, event placement, I/O binding, and allocator-backed buffers twice. The first run uses typed `disable_synchronize_execution_providers=1` options and observes `cudaErrorNotReady`; the second uses default `Ort::RunOptions` and observes a ready event. Both ordinary graph runs continue to use typed asynchronous options, and the proof symbol is absent from the production native archive.
- Every caller-supplied GPU library now requires `staged_sha256` in addition to source size and SHA-256. Expected staged inventory is built independently from those caller hashes, verified pinned ORT archive contents after the deterministic RUNPATH transform, and TensorRT alias hashes bound to their versioned caller entries. The exact caller-manifest digest remains the generation key. Reuse compares the exact directory inventory and every file hash with this trusted inventory; `.integrity` contains only the atomic completeness marker. Tests accept valid reuse and reject a forged file plus forged seal, a deleted file, an added file, and an incorrect expected staged hash.
- `enter_device_for_test` and `inject_backend_failure_for_test` now catch unknown exceptions and return typed inference status instead of allowing termination. A test-double-only boundary throws a non-`std::exception` value and proves Rust receives the expected typed error. The structured-status boundary audit found exhaustive native, standard, and unknown catches on the remaining CXX wrappers.

## Final Three-Finding Verification

- `cargo test --workspace`: pass.
- `cargo test --workspace --features native-test-double`: pass, including the non-standard exception containment test and all nine GPU build-contract tests.
- `cargo test --workspace --features onnx-cuda,tensorrt,native-hardware-tests`: pass with the typed staged-hash manifest.
- `cargo test --test stub_gpu --features onnx-cuda,tensorrt,native-hardware-tests -- --ignored --nocapture`: 6 passed on the real GPU, including the durable async/default negative control.
- `cargo test --test multigpu --features onnx-cuda,tensorrt,native-hardware-tests -- --ignored --nocapture`: pass with the explicit one-GPU skip.
- Strict Clippy passed for default, native test double, and combined ONNX CUDA/TensorRT/hardware configurations. Formatting and `git diff --check` passed.
- Production `nm` found no hardware-proof or test-mutation symbols. Source and folder limits pass, CMake still declares one native library and no executable, and neither `tinfer_rust/espeak_align` nor `tools/styletts2_model_scripts` changed.

No open implementation concern remains. This host still has one physical GPU, so the real two-GPU test retains its explicit skip while the deterministic two-device test remains green.

## Symlink-Backed Inventory Remediation

- Manifest validation now derives each logical staged library name from the validated strict-relative manifest path before resolving symlinks. The resolved target remains canonicalized and root-contained, and its retained descriptor, device/inode identity, source size, source SHA-256, ELF contract, and symbols remain the trusted copy source.
- Logical-name and canonical-target duplicate checks are separate and explicit. Non-normal relative components and root-escaping symlinks fail before staging.
- Standard TensorRT entries such as `libnvinfer.so.11 -> libnvinfer.so.11.1.0` retain `libnvinfer.so.11` through required-name validation and staging. Linker aliases are generated from the staged ABI name, and their expected hashes remain bound to the manifest-provided post-transform `staged_sha256`.
- CUDA ABI-name symlinks likewise satisfy the typed required inventory without callers creating renamed regular copies. Staging reads only from the retained canonical-target descriptor and publishes under the manifest alias.

### TDD and Verification

- RED: the symlink-backed TensorRT and CUDA tests failed because canonical target basenames replaced manifest ABI names; duplicate logical aliases were accepted.
- GREEN: `cargo test --test gpu_build_contract --test gpu_manifest_aliases` passes 14 contracts. The five alias contracts cover TensorRT required names plus linker alias generation and sealing, all required ONNX-CUDA CUDA/cuDNN names and sealing, unsafe relative components, root escape, and duplicate logical aliases. The original descriptor tests still reject target replacement and synchronized in-place mutation.
- `cargo test --workspace` and `cargo test --workspace --features native-test-double`: pass.
- Strict workspace/all-target Clippy with warnings denied passes for default and `native-test-double`; both Rust workspaces pass formatting.
- `git diff --check`, changed-file limits, folder limits, and protected `espeak_align` / `tools/styletts2_model_scripts` checks pass.
- Real multi-gigabyte GPU acquisition and hardware execution were not rerun because no native model/CUDA source changed. The changed production build-helper path is exercised by the focused contracts and both default/test-double compile gates.

No open concern remains for this remediation.

## Canonical-Target Alias Regression

- Added a real-symlink manifest contract where the distinct safe ABI basenames
  `libcublas.so.12` and `libcublasLt.so.12` resolve to one canonical regular
  library. Both entries carry valid typed CUDA metadata, distinct logical names,
  valid ELF symbols, and source/staged integrity hashes.
- The regression asserts the exact `GPU library canonical target is duplicated`
  error, proving validation passes the logical-name check and reaches the
  canonical-target duplicate check.
- `cargo test --test gpu_manifest_aliases --test gpu_build_contract`: 15 passed.
- `cargo fmt --all -- --check`, `git diff --check`, file/folder limits, and
  protected `espeak_align` / `tools/styletts2_model_scripts` checks pass.
