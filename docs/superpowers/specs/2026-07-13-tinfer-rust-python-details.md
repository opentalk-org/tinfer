# Tinfer Rust Python Compatibility Details

## Compatibility Boundary

The Rust port ships a Python distribution named `tinfer`. It is a frontend to
the in-process Rust library, not a retained Python engine. Compatibility means
code written against the current public Python package continues to import and
observe the same contract.

The parity boundary includes names without a leading underscore in the current
package modules, names listed in `__all__`, generated protobuf modules, and the
documented server classes. Private worker, shared-memory, scheduler, and model
implementation modules are not exported because their replacement is the
purpose of the port.

The following properties must match:

- import path and exported name;
- callable kind, positional/keyword parameters, annotations, and defaults;
- synchronous method, coroutine, iterator, or async-generator behavior;
- enum values, dataclass fields, mutability, equality, and representations;
- return Python types, NumPy dtype/shape, ordering, and ownership;
- raised exception class and stable message for validation failures;
- cancellation, close, repeated-close, and object-lifetime behavior.

Parity does not mean preserving implementation-only attributes such as
`executor`, `_workers`, queues, locks, or multiprocessing state.

## Package and Build Layout

```text
tinfer_rust/
├── Cargo.toml
├── pyproject.toml
├── src/lib.rs
├── src/python/          PyO3 bindings and conversions
└── python/tinfer/
    ├── config/          Python dataclass facade
    ├── core/            engine, async engine, stream, request
    ├── server/grpc/     lifecycle facade and generated stubs
    ├── server/websocket/ aiohttp compatibility transport
    ├── server/health.py
    ├── support/
    └── _native.*         one PyO3 extension
```

Maturin builds the same Rust library crate as a `cdylib`. The executable and
extension link the same unified model-native static target; the wheel contains
no C++ executable or inference sidecar. Cargo feature `python` gates PyO3,
NumPy, and asyncio adapters without changing core types.

`src/lib.rs` owns configuration, engine construction, scheduling, models,
audio, and service factories. `main.rs` only parses process configuration and
calls the library. Python never launches `main.rs` and never uses localhost RPC
to reach the engine.

## Public Surface Inventory

The initial checked manifest contains these current entry points:

| Module | Required public surface |
| --- | --- |
| `tinfer.config.engine_config` | `StreamingTTSConfig`, `from_yaml`, `to_yaml` |
| `tinfer.core.request` | `AlignmentType`, `StreamParams`, `ModelInfo`, `VoiceInfo`, `AlignmentItem`, `Alignment`, `AudioChunk` |
| `tinfer.core.engine` | `StreamingTTS` and all its current public methods |
| `tinfer.core.async_engine` | `AsyncStreamingTTS` and all its current public methods |
| `tinfer.core.stream` | `TTSStream` and all its current public methods |
| `tinfer.server.health` | `HealthState`, properties, admission, drain, and stop methods |
| `tinfer.server.grpc.server` | `GRPCServer`, `start`, `stop`, `serve` |
| `tinfer.server.grpc` | `styletts_pb2`, `styletts_pb2_grpc` |
| `tinfer.server.websocket` | `WebSocketServer`, `WebSocketHandler` |
| `tinfer.support` | `InferenceError` and current observability exports |

The manifest records full `inspect.signature` output and structured definitions
rather than relying on this summary. A deliberate public API change requires
updating the source Python package, manifest, Rust facade, and parity fixture in
one reviewed change.

## Python-Owned Value Types

The thin package defines `AlignmentType` with the current string values,
`StreamParams` as a `TypedDict`, and the current dataclasses with identical
field order and defaults. Keeping these in ordinary Python preserves dataclass
introspection, enum identity, pickling, and typing behavior more exactly than
pretending PyO3 classes are dataclasses.

Native conversion functions accept those objects, validate fields once, and
construct typed Rust requests. Results are converted back to the Python
dataclasses at the boundary. No raw dictionary crosses into scheduler or model
code.

`AudioChunk.audio` is a one-dimensional contiguous NumPy array. Float engine
audio is exposed as `float32`; already encoded audio uses the current byte
dtype. Rust transfers an owned buffer into NumPy where supported and otherwise
performs one boundary copy. Native execution slots are released before Python
receives an array, so array lifetime never pins CUDA buffers or a scheduler
lease.

## Configuration Mapping

`StreamingTTSConfig` retains the current Python field names, defaults,
`from_yaml`, and `to_yaml` serialization. The facade converts it to the new
higher-level Rust engine configuration:

- `devices` becomes the automatic-placement device allowlist;
- `batch_size_per_device` and `default_batch_size` become explicit placement
  batch limits emitted by automatic placement;
- `process_workers_per_gpu` becomes replicas per selected GPU;
- `runtime_engine` selects ONNX Runtime or TensorRT;
- `compile_models` is accepted for signature parity but compilation is an
  exporter concern and therefore cannot trigger request-time compilation;
- `executor_type="process"` is accepted as the sole current enum value but does
  not create processes.

Invalid combinations fail during engine creation. The compatibility facade
does not silently change backend, platform, or device.

## Engine and Stream Bindings

`StreamingTTS` holds an `Arc<Engine>` plus a Rust Tokio runtime handle. Methods
that wait synchronously release the GIL while sending commands to the scheduler
actor. Nonblocking catalog methods read immutable snapshots without entering
Python callbacks.

`AsyncStreamingTTS` wraps the same engine handle. PyO3 asyncio integration
converts Rust futures into futures bound to the caller's running event loop.
Completion is scheduled back onto that loop; Rust worker threads never call
arbitrary Python code.

`TTSStream` wraps a Rust stream ID and receiver:

- `add_text`, `force_generate`, `try_generate`, `cancel`, `get_state`, and
  `close` preserve their current callable kinds;
- `wait_for_audio` returns one awaitable chunk;
- `pull_audio` is a Python async generator over a native `_next_audio` future;
- `get_audio` drains only results already available;
- `collect_audio` releases the GIL while waiting for admitted chunks.

The wrapper closes exactly once. Finalization sends best-effort cancellation,
but correctness and slot release do not depend on Python garbage collection.
Cancellation generation checks remain in the Rust scheduler.

`generate`, `generate_full`, and `generate_full_batch` use these same stream
commands. Audio merging and alignment offsetting are implemented once in Rust
and tested against current NumPy results.

## Model Loading and Registration

`load_model(model_id, model_path, voices_folder=None)` resolves the artifact
manifest and submits automatic placements. Paths are passed losslessly and
catalog changes become visible only after every selected placement is warm.

`register_model(model_id, model, device=None, keep_in_main=True)` retains its
current signature and validation that the object is loaded. It invokes the
owning model module's Python registration importer exactly once to turn
`model.get_state()` into the same typed artifact representation accepted by
`load_model`. The importer lives with that model, and its output is handed to
Rust before replicas are created.

Inference never calls the registered Python object. `keep_in_main=False`
applies the current object-state release behavior after successful import. A
model without a registration importer raises a typed `ValueError`; it is not
adapted through a Python inference callback. The deterministic stub includes a
small importer fixture so this path is testable before a real model is shipped.

## Server Bindings

`GRPCServer` and `WebSocketServer` are Python lifecycle facades over Rust
service handles. `start`, `stop(grace_period=5.0)`, and `serve` retain their
coroutine signatures. Starting either server does not construct another engine
or CUDA runtime.

Python protobuf files are generated from the same `styletts.proto` during wheel
build. Their package import paths remain unchanged, allowing current Python
gRPC clients to connect to the Rust Tonic server.

`WebSocketServer.create_app()` remains available for current aiohttp test and
embedding code. It creates an `aiohttp.web.Application` whose thin handlers
translate HTTP bodies and WebSocket frames into a Rust service/session handle.
Request validation, route semantics, WebSocket state transitions, scheduling,
and response formatting remain Rust-owned. The adapter owns transport objects
only and is never used by the standalone Rust server.

`WebSocketHandler` similarly wraps a Rust single-context session for callers
that currently instantiate it. Frame receipt/send stays in aiohttp; each
decoded command and each outgoing typed event crosses the binding. Multi-stream
state is not duplicated in Python.

`HealthState` wraps the Rust health actor. Python properties read one coherent
snapshot, while admission and drain methods await actor commands. Python and
Rust server facades sharing an engine also share its health state.

## Errors, GIL, and Runtime Lifetime

Rust errors map explicitly to current Python exception categories. Request
validation uses `ValueError`; invalid lifecycle use uses `RuntimeError`;
inference failures use `tinfer.support.InferenceError`. Cancellation maps to
`asyncio.CancelledError` only when the Python awaitable itself is cancelled.

Dropping a Python awaitable cancels its receiver and sends a scheduler
cancellation command. Native work already submitted may finish, but generation
checks discard late output. Extension teardown drains Python-facing tasks,
stops service handles, shuts down the engine, destroys native pipelines, then
releases CUDA primary contexts. No PyObject is retained by a native pipeline.

The GIL is held only for argument extraction, Python value construction,
one-time registration import, and aiohttp transport adaptation. Scheduling,
chunking, preprocessing, inference, postprocessing, encoding, and catalog
operations execute without it.

## Parity Generation and Tests

A repository script imports the current Python reference in an isolated
environment and writes a reviewed JSON manifest containing modules, exports,
signatures, enum members, dataclass fields/defaults, and protobuf descriptors.
It rejects unrepresentable defaults instead of serializing `repr` guesses.

The same black-box suite runs once against the reference package and once
against the Rust-backed wheel. It covers:

- import and `inspect.signature` equality;
- config YAML round trips and validation errors;
- dataclass construction, mutation, equality, enum, and NumPy contracts;
- sync and async engine lifecycle, catalogs, loading, and registration;
- stream triggers, timeouts, ordering, cancellation, collection, and close;
- async-generator cancellation and event-loop affinity;
- gRPC generated-client calls and server lifecycle;
- aiohttp `create_app`, HTTP routes, and both WebSocket modes;
- repeated startup/shutdown and Python interpreter teardown;
- concurrent streams across multiple GPUs with stable device-runtime identity.

Wheel CI builds CPython 3.11+ artifacts for supported Linux targets, installs
each wheel into an empty environment, and runs the suite without the source
tree on `PYTHONPATH`. CPU tests are mandatory; CUDA and TensorRT parity tests
run only on matching workers and never substitute a CPU fallback.
