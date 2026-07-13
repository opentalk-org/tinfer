# Tinfer Rust Port Design

## Objective

Create `tinfer_rust/` as the production implementation of Tinfer's engine,
scheduling, model runtime, gRPC service, and ElevenLabs-compatible HTTP and
WebSocket service. Preserve the behavior of the current working tree while
removing Python multiprocessing and request-time Python model execution from
the serving path.

The first model uses deterministic stub artifacts from the future external
exporter; PyTorch architecture, training, conversion, and export are not ported.

`tools/styletts2_model_scripts/` remains in the repository and is not removed
or rewritten by this work.

## Current Runtime Being Ported

The design preserves these current behaviors rather than mirroring Python file
names:

- `StreamingTTS` owns streams, model lifecycle, warmup, timeout wakeups, chunk
  merging, catalog access, and shutdown.
- `SchedulingMixin` maintains per-stream text buffers, derived chunk limits,
  source spans, forced generation, cancellation epochs, and result ordering.
- `WorkerScheduler` groups queued work by model, caps batches per device, ages
  requests, and boosts requests whose buffered audio is exhausted.
- `ProcessExecutor`, `WorkerManager`, shared-memory serialization, and spawned
  workers provide Python/GPU isolation. They are replaced by in-process Rust
  device runtimes.
- StyleTTS2 preprocessing preserves original spans through normalization,
  espeak alignment, vocabulary filtering, tokenization, and long-text windows.
- Model context carries voice conditioning and the previous style vector across
  sequential chunks.
- Model postprocessing trims output, produces phoneme/word/character timing,
  preserves source spans, and merges long-text windows.
- gRPC exposes health, models, voices, unary synthesis, server-streaming
  synthesis, and bidirectional incremental synthesis.
- HTTP/WebSocket exposes health, discovery, four HTTP synthesis routes, and
  single- and multi-context incremental WebSocket routes with strict state
  transitions.

Rust and Python protobuf bindings are generated from the existing protobuf
contract; generated files are not hand-ported.

## Selected Architecture

Use one modular Rust application crate with a separate copied `espeak_align`
Cargo project. All serving components run in one process.

```text
tinfer_rust/
├── Cargo.toml
├── build.rs
├── proto/
├── native/
├── python/
├── src/
│   ├── config/
│   ├── core/
│   ├── engine/
│   ├── scheduling/
│   ├── models/
│   ├── audio/
│   ├── server/
│   ├── support/
│   └── main.rs
├── tests/
└── espeak_align/
```

Each directory stays below 16 files and every source file below 300 lines.

### Core and engine placement

Core accepts only explicit placements: model identity, artifact directory,
backend, platform implementation, device, replica count, batch limit, and
execution-slot count. It never guesses a device.

The higher-level engine provides automatic placement. It inspects configured
devices and model requirements, constructs explicit placements, and submits
them to core. Explicit public configuration bypasses automatic selection.

### Device runtime

The process creates one long-lived device runtime per configured GPU. Each
runtime retains the CUDA primary context once and owns its scheduling queue.
Model replicas, TensorRT execution contexts, ONNX Runtime sessions, CUDA
streams, and device buffers attach to that runtime.

No request, stream, model replica, or protocol connection creates a CUDA driver
context. Multiple execution slots provide concurrency within the retained
context. The global scheduler dispatches compatible batches to the least-loaded
eligible replica across GPUs.

CPU preprocessing, CPU-native stages, resampling, and encoding run on bounded
executors so they cannot block Tokio's asynchronous I/O workers.

## Model Ownership and Native Build

All model-specific code is contained in `src/models/<model_name>/`. A model may
contain Rust preprocessing and postprocessing plus native implementation under
`cpp/`.

```text
src/models/stub/
├── mod.rs
├── config.rs
├── model.rs
├── preprocessing.rs
├── postprocessing.rs
├── artifacts/
└── cpp/
    ├── include/
    ├── pipeline/
    ├── onnx/
    ├── tensorrt/
    ├── cpu/
    ├── cuda/
    └── vendor/
```

The root Rust build compiles every `src/models/*/cpp/` tree into one native
library linked into the Rust server. There are no C++ executables, per-model
binaries, inference subprocesses, or native helper daemons.

Generic native code is limited to the safe bridge, device-runtime ownership,
status types, and shared RAII utilities. Model tensor contracts, stage order,
backend bindings, platform operations, and vendor-specific code remain inside
the owning model directory.

Rust accesses native pipelines through safe model-specific wrappers over opaque
C++ handles. C++ exceptions never cross the bridge. Native allocations,
sessions, execution contexts, streams, and primary-context references use RAII.
Rust request buffers remain alive for the entire native call and returned audio
is copied into Rust-owned memory before the execution slot is released.

## Model Pipeline Contract

The engine sees a typed `Model` trait with model metadata, placement
requirements, warmup, batch inference, voice discovery, and shutdown. It does
not see ONNX, TensorRT, CPU, CUDA, or vendor types.

Each model's C++ `pipeline/` explicitly glues its ordered stages. The stub proves
the required alternating shape:

```text
preprocessing
  -> ONNX Runtime or TensorRT graph
  -> CPU or CUDA fast-processing stage
  -> ONNX Runtime or TensorRT graph
  -> postprocessing
```

The sequence is model code, not a global hard-coded pipeline. A future model may
use a different number and order of graph and native stages without changing
the engine.

ONNX Runtime supports CPU and CUDA execution providers. TensorRT placements are
CUDA/NVIDIA-only. Invalid backend/platform/device combinations fail while the
placement is loaded, before the server becomes ready.

The stub artifact manifest records model metadata, languages, sample rate,
voices, graph files, supported backends, supported platforms, tensor contracts,
and warmup shapes. Tests use small deterministic ONNX graphs and matching
TensorRT plans when TensorRT hardware is available.

## Scheduling and Concurrency

One Rust scheduler actor owns stream readiness, timeouts, cancellation,
compatible batching, priority, placement selection, slot leases, context
updates, and ordered delivery. Focused source files do not create independent
schedulers or state owners. CPU, native, and encoding jobs run on bounded
executors and return typed completion commands to the actor.

Requests from one stream remain sequential when model context depends on the
previous chunk. Independent streams execute concurrently. Batches contain one
model, backend, device-compatible shape class, and parameter compatibility
class. Cancellation invalidates queued and late results using a monotonically
increasing generation rather than Python UUID nonces.

Backpressure is bounded at protocol admission, stream queues, device queues,
and response encoders. Queue capacity is explicit configuration; overload is a
typed error rather than unbounded allocation.

## espeak_align

Copy the current `tinfer/espeak_align` Cargo workspace into
`tinfer_rust/espeak_align` and preserve it as a separate Cargo project. The Rust
model preprocessing path calls `espeak-align-core` directly. Its alignment,
span, punctuation, batch, worker-pool, and language behavior is covered by the
current alignment fixtures.

The copied PyO3 wrapper is independent of the server and main Python frontend.

## Python API Compatibility

The crate builds a PyO3 `cdylib`, and thin `python/tinfer/` modules preserve the
current public imports and behavior through the same in-process Rust engine.
There is no executable, RPC sidecar, second inference library, or request-time
Python model callback. Exact signature, type, async, NumPy, server, protobuf,
registration-import, and `aiohttp` adapter contracts are defined and tested by
the Python companion spec.

## Protocol and Audio Boundaries

Use Tokio for asynchronous orchestration, Tonic for gRPC, and Axum for HTTP and
WebSocket. The protobuf source remains the wire authority.

Protocol handlers translate typed wire requests into engine commands and
translate engine events into wire responses. They do not schedule devices or
call native pipelines directly.

Audio remains float mono PCM inside the engine. PCM, μ-law, A-law, WAV, MP3,
and Opus encoding occurs at the protocol edge. Compressed streaming uses one
response-scoped encoder so the complete byte stream is decodable. Character
alignment remains structured data until response formatting.

Single-context and multi-context WebSockets are explicit state machines. They
preserve initialization, immutable settings, context reinitialization,
keepalive, conditional trigger, flush, finalization, inactivity expiry,
disconnect cleanup, and exactly-once final messages from the current contract.

## Health, Lifecycle, and Errors

Readiness requires all configured placements to load and warm successfully.
Liveness remains true until shutdown completes. Draining rejects new synthesis
admission, waits for admitted transports, then stops protocol servers, engine
queues, model replicas, and device runtimes in that order.

Use typed Rust errors for configuration, catalog resolution, validation,
overload, cancellation, native loading, inference, encoding, and shutdown.
Native errors include model, stage, backend, platform, and device context.
Protocol adapters map errors to their existing gRPC status, HTTP body/status,
or WebSocket error/close contract.

An inference failure affects its batch and requests but does not silently return
empty successful audio. A failed placement becomes unhealthy and stops receiving
work. Recoverable backend failures reload inside the retained device runtime.
Fatal CUDA device failures disable that device and all its placements rather
than resetting a shared context underneath other models.

## Verification Strategy

- Port current chunk schedule, span, cancellation, ordering, and starvation
  cases as Rust unit tests before implementing the scheduler.
- Reuse current espeak alignment strings and spans as cross-language fixtures.
- Use a pure Rust fake model to test engine lifecycle and scheduling without a
  native dependency.
- Test the stub pipeline stage by stage, then compare ONNX CPU, ONNX CUDA, and
  TensorRT outputs within declared tolerances.
- Stress one GPU with concurrent streams and assert the CUDA primary-context
  retain count and device-runtime identity remain stable.
- Test explicit placement in core and deterministic automatic placement in the
  engine across simulated heterogeneous devices.
- Generate Rust protobuf bindings and run unary, streaming, incremental,
  discovery, and health integration tests.
- Port all current HTTP/WebSocket contract and state-machine tests, including
  error timing, admission accounting, audio formats, and cleanup ordering.
- Run CPU-only CI for core, protocols, ONNX CPU, and espeak. Gate CUDA, ONNX CUDA,
  and TensorRT suites on matching hardware and libraries.
- Treat Python/Rust black-box parity fixtures as the cutover gate while the
  Python implementation remains available as a reference.

## Delivery Increments

1. Establish the Rust crate, build system, typed contracts, and parity fixtures.
2. Port text chunking, stream state, timeout triggering, and stream scheduling.
3. Copy and integrate `espeak_align`; port span-preserving preprocessing and
   alignment conversion.
4. Build the unified native library and deterministic stub pipeline on ONNX CPU.
5. Add the shared CUDA device runtime, ONNX CUDA, CUDA stages, and TensorRT.
6. Implement explicit core placement, automatic engine placement, replicas,
   batching, backpressure, cancellation, warmup, and multi-GPU dispatch.
7. Port audio processing and response-scoped encoders.
8. Port gRPC health, catalog, unary, streaming, and incremental services.
9. Port HTTP discovery and the four synthesis routes.
10. Port single- and multi-context WebSocket state machines.
11. Build the PyO3 package and satisfy the Python API and behavior parity suite.
12. Add observability, graceful draining, deployment configuration, stress
    tests, parity gates, and production cutover documentation.

## Detailed Implementation Contracts

- [Runtime and native details](2026-07-13-tinfer-rust-runtime-details.md)
- [Protocol and operations details](2026-07-13-tinfer-rust-protocol-details.md)
- [Python compatibility details](2026-07-13-tinfer-rust-python-details.md)

## Explicit Non-Goals

- Porting PyTorch model architecture or training code to Rust
- Porting model export or checkpoint conversion into the serving crate
- Removing or rewriting `tools/styletts2_model_scripts/`
- Shipping a real StyleTTS2 model in the stub milestone
- C++ executables, inference subprocesses, or per-model native binaries
- Dynamic model plugins or a stable third-party plugin ABI
- Python multiprocessing, Python inference callbacks, or RPC to a sidecar
