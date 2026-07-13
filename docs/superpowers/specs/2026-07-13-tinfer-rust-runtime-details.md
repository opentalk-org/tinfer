# Tinfer Rust Runtime Implementation Details

## Module Layout

```text
tinfer_rust/src/
├── config/       mod.rs, engine.rs, model.rs, server.rs
├── core/         mod.rs, request.rs, stream.rs, chunk.rs, alignment.rs
├── scheduling/   mod.rs, scheduler.rs, state.rs, queue.rs, priority.rs, timeout.rs
├── engine/       mod.rs, engine.rs, catalog.rs, placement.rs, lifecycle.rs
├── models/       mod.rs, traits.rs, registry.rs, stub/
├── audio/        mod.rs, format.rs, resample.rs, pcm.rs, compressed.rs
├── server/       mod.rs, health.rs, grpc/, http/, websocket/
├── support/      mod.rs, error.rs, telemetry.rs, shutdown.rs
└── main.rs
```

Every folder stays below 16 direct files and every source below 300 lines.
`src/models/stub/` owns its Rust, artifacts, tests, and `cpp/` subtree.

## Typed Runtime Contracts

The untyped Python dictionaries become concrete Rust types. Core identifiers
use newtypes so model, voice, request, stream, replica, and device identifiers
cannot be mixed.

```rust
pub struct ModelId(pub Arc<str>);
pub struct VoiceId(pub Arc<str>);
pub struct RequestId(pub Uuid);
pub struct StreamId(pub Uuid);
pub struct ReplicaId(pub u32);
pub struct DeviceId(pub u32);

pub enum AlignmentKind { Word, Character, Phoneme, None }
pub enum BackendKind { Onnx, TensorRt }
pub enum PlatformKind { Cpu, Cuda }

pub struct StreamOptions {
    pub chunk_schedule: NonEmptyVec<usize>,
    pub timeout: Duration,
    pub alignment: AlignmentKind,
    pub output: AudioOutput,
    pub model: ModelParameters,
}
```

`NonEmptyVec` validates positive, nondecreasing sizes once. Model parameters
are a model-owned enum variant rather than a raw map. The stub uses
`ModelParameters::Stub(StubParameters)`; a future StyleTTS2 module owns its
typed parameter struct.

`AudioChunk` owns `Arc<[f32]>` mono samples inside the engine, a sample rate,
chunk index, source `Range<usize>`, optional alignment, and request generation.
Encoded bytes exist only in protocol/audio modules.

## Engine and Stream API

The public engine returns an owned stream backed by bounded Tokio channels.
Dropping it sends cancellation and closes its receiver.

```rust
impl Engine {
    pub async fn load_model(&self, request: LoadModel) -> Result<ModelMetadata>;
    pub async fn unload_model(&self, model: &ModelId) -> Result<()>;
    pub async fn create_stream(&self, request: CreateStream) -> Result<StreamHandle>;
    pub fn models(&self) -> Arc<[ModelMetadata]>;
    pub fn voices(&self, model: &ModelId) -> Result<Arc<[VoiceMetadata]>>;
}

impl StreamHandle {
    pub async fn append(&self, text: String) -> Result<()>;
    pub async fn trigger(&self) -> Result<()>;
    pub async fn flush(&self) -> Result<()>;
    pub async fn cancel(&self) -> Result<()>;
    pub async fn recv(&mut self) -> Option<Result<AudioChunk>>;
}
```

`trigger` is conditional readiness; `flush` is unconditional generation. The
engine actor owns stream state, so protocol tasks cannot mutate scheduling
fields directly or clear another task's queue.

## Model Interfaces

The engine-facing interfaces are synchronous at the replica boundary because
native execution runs on bounded device executors. Tokio-facing orchestration
wraps these calls in scheduled jobs.

```rust
pub trait ModelFactory: Send + Sync {
    fn metadata(&self) -> &ModelMetadata;
    fn requirements(&self) -> &PlacementRequirements;
    fn load(&self, placement: &Placement) -> Result<Box<dyn ModelReplica>>;
}

pub trait ModelReplica: Send + Sync {
    fn slot_count(&self) -> usize;
    fn warmup(&self, slot: ExecutionSlotId, shapes: &[WarmupShape]) -> Result<()>;
    fn infer(&self, slot: ExecutionSlotId, batch: PreparedBatch) -> Result<CompletedBatch>;
    fn voices(&self) -> &[VoiceMetadata];
}
```

`PreparedBatch` contains requests for one compatible model placement. It owns
flattened typed buffers plus offsets, lengths, request generations, and model
context. `CompletedBatch` returns one result per input in the same order. A
length mismatch is an invariant error and fails the whole batch.

The compile-time registry maps a model kind to its factory constructor. Adding
a model requires adding its module and one typed registry entry; no dynamic
plugin ABI or string-based class import exists.

## Artifact Manifest

Every deployment model directory contains `manifest.toml`, graph artifacts,
and voice metadata. The runtime deserializes it into a model-owned manifest and
validates it before creating a placement.

```text
model/
├── manifest.toml
├── graphs/graph_1.onnx
├── graphs/graph_2.onnx
├── tensorrt/<target>/graph_1.plan
├── tensorrt/<target>/graph_2.plan
└── voices/*.bin
```

The manifest declares model kind and revision, sample rate, languages, default
language, voice files, graph names, supported backend/platform pairs, tensor
dtypes and ranks, batch and sequence bounds, warmup shapes, and TensorRT build
target metadata. Missing artifacts, incompatible TensorRT/CUDA versions,
unsupported compute capability, invalid tensor contracts, and duplicate voices
fail placement loading.

## Placement Implementation

Core accepts this explicit configuration:

```rust
pub struct Placement {
    pub model: ModelId,
    pub artifact_dir: PathBuf,
    pub backend: BackendKind,
    pub platform: PlatformKind,
    pub device: ExecutionDevice,
    pub replicas: NonZeroUsize,
    pub slots_per_replica: NonZeroUsize,
    pub max_batch: NonZeroUsize,
}
```

`ExecutionDevice` is `Cpu` or `Cuda(DeviceId)`. Core rejects TensorRT on CPU,
CUDA stages on CPU, devices absent from inventory, and placements outside the
manifest limits.

Automatic engine placement receives model requirements, device inventory, and
requested replica count. It filters incompatible devices, sorts GPUs by free
memory then stable device ID, distributes replicas round-robin across eligible
GPUs, and emits explicit `Placement` values. Explicit user placements bypass
this policy. Tests use a fake inventory so results are deterministic.

## Device Runtime and CUDA Lifetime

The native library contains one process-wide `DeviceRegistry`. For every CUDA
device used by an explicit placement it calls `cuDevicePrimaryCtxRetain` once,
stores the returned `CUcontext`, and releases it only during engine shutdown.
A reference-counted `DeviceRuntime` is shared by every model pipeline on that
device.

Each model replica owns a fixed pool of execution slots. A TensorRT slot owns
one `IExecutionContext`, one CUDA stream, reusable device buffers, and completion
events per graph stage. An ONNX CUDA slot owns a session configured with its
user compute stream, its I/O bindings, and reusable buffers. ONNX CPU slots own
independent bindings and share the immutable session environment.

Before native work on an OS thread, a scoped context guard pushes the retained
primary context and pops it afterward. This changes the thread's current
context; it never creates another context.

The Rust device executor leases an available slot, runs its blocking bridge
call on the bounded device pool, and returns the slot only after native
completion. Multiple leased slots use different streams and execute
concurrently within the same primary context.

## Unified Native Build and Bridge

`build.rs` invokes one CMake build. The CMake target explicitly includes generic
bridge/device sources and every registered `src/models/*/cpp/` source list. It
compiles one static `tinfer_models_native` library and links ONNX Runtime,
TensorRT, CUDA Driver, and CUDA Runtime once into the Rust server.

Each model defines a `cxx::bridge` module and a model-prefixed opaque pipeline
type. Shared bridge structs contain only fixed-width scalars, enums, strings,
and flat vectors. Nested tensors use flat storage with typed offsets and shapes.
C++ exceptions are caught and translated to `Result`; no exception or raw
native pointer crosses into ordinary Rust code.

`build.rs` first runs `cxxbridge` for every registered model bridge and passes
the generated C++ sources to that same CMake target. The native API exposes
create, warmup, infer, and close through the opaque pipeline. Construction
receives a shared device-runtime handle, artifact paths, backend/platform
selection, slot count, and validated shape bounds.

## Stub Model Pipeline

Rust preprocessing normalizes input, converts it into deterministic integer
features, resolves the stub voice, creates request-scoped seeded values, and
builds a flat native batch. Rust postprocessing converts native output and
durations into `AudioChunk`, alignment, and next model context.

The model-owned C++ pipeline implements:

```text
graph_1(ONNX Runtime | TensorRT)
  -> fast_stage(CPU | CUDA)
  -> graph_2(ONNX Runtime | TensorRT)
```

`pipeline/` owns this order and the typed intermediate structures. `onnx/` and
`tensorrt/` implement the two graph interfaces. `cpu/` and `cuda/` implement the
same fast-stage interface. Backend and platform selection occurs once during
pipeline construction, not inside every request.

Graph 1 produces a deterministic feature tensor and per-token durations. The
fast stage expands features according to durations. Graph 2 converts expanded
features to deterministic float audio. This exercises dynamic batch and
sequence shapes, alternating graph/native stages, result splitting, and timing
without pretending to be StyleTTS2.

## Unified Scheduler

One `Scheduler` actor owns every mutable scheduling structure. Protocol and
engine tasks send bounded commands: create stream, append text, trigger, flush,
cancel, close, timeout, placement change, and executor completion. The actor
mutates state, selects ready work, leases slots, and dispatches owned jobs. It
does not await CPU or native work while processing a command.

`StreamState` owns the full source text, committed byte offset, chunk index,
prepared chunks, pending count, generation-window deadline, model context, and
cancellation generation. Text indices are UTF-8 byte ranges internally and are
converted to character indices at protocol formatting boundaries.

The chunker ports the current derived limits exactly. It preserves separator
whitespace, uses sentence then punctuation then word boundaries, falls back to
derived hard limits, and records source ranges before dispatch. Golden fixtures
cover abbreviations, Unicode punctuation, repeated substrings, leading spaces,
and long input.

Timeouts use one Tokio `DelayQueue` entry per active generation window. Adding
text to an empty pending window inserts a deadline. Further text retains that
deadline. Dispatch removes it; remaining text receives a fresh deadline. Flush
sets unconditional readiness. Conditional trigger only evaluates ordinary
readiness.

Only one contextual chunk from a stream may be pending. Its completion updates
model context before the next prepared chunk is eligible. Cancellation clears
buffers and queues, increments the generation, and drops results carrying an
older generation.

Jobs are grouped by model, placement, backend, platform, shape bucket, and model
parameter compatibility. Within a group, priority is request age plus a
starvation boost when elapsed playback time exceeds collected audio duration.
The scheduler takes at most the placement batch limit and chooses the eligible
replica with the lowest tuple of queued work, leased slots, and stable IDs.

Each device queue and stream output channel has an explicit bounded capacity.
Admission beyond capacity returns `Overloaded`; it never silently drops work or
allocates an unbounded queue. Completed batches are split by request, checked
against cancellation generation, and delivered in stream chunk order.

CPU preprocessing and native inference receive immutable owned jobs through
bounded executor queues. Completion messages carry the slot, batch outputs,
request generations, and timing. The scheduler releases the slot, applies model
context, queues output, and immediately reevaluates all newly eligible work.

## Shutdown and Recovery

Engine shutdown stops new jobs, cancels timeout entries, drains or cancels
queued jobs according to the configured grace period, waits for leased slots,
destroys pipelines, destroys execution slots, and finally releases retained
primary contexts.

A model-stage error fails the affected batch with typed per-request errors. A
recoverable session or engine failure marks one placement unavailable and lets
the engine reconstruct it on the same `DeviceRuntime`. A fatal CUDA error marks
the device and every placement on it unavailable. The runtime does not reset the
shared primary context automatically because that would invalidate other model
objects. Other GPUs and CPU placements continue serving; restoring the failed
device requires a controlled server restart.

## Rust and Native Dependencies

The runtime uses Tokio primitives, `uuid`, `serde`, `serde_yaml`, `toml`,
`thiserror`, `tracing`, `cxx`, and `cmake`. Native compilation uses C++20, CUDA,
ONNX Runtime's C++ API, and TensorRT's C++ API. Optional backend support is a
compile-time Cargo feature; a configured placement whose feature is absent is a
configuration error rather than a runtime fallback.
