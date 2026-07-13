# Tinfer Rust Protocol and Operations Implementation Details

## Server Composition

`main.rs` loads typed YAML configuration, initializes telemetry, discovers
devices, resolves automatic and explicit model placements, loads and warms the
engine, marks readiness, and starts gRPC and HTTP/WebSocket listeners on the
same Tokio runtime.

Tonic owns the gRPC listener. Axum owns HTTP and WebSocket routes. Both receive
clones of one `Arc<Engine>` and one `Arc<HealthState>`. Protocol handlers never
hold model replicas or native handles.

Protocol dependencies are `tonic`, `prost`, `tonic-health`, `axum`,
`tokio-stream`, `tower`, `tower-http`, `serde`, `serde_json`, `tracing`, and the
OpenTelemetry tracing layer. Audio uses `rubato` for stateful resampling and
supervises the repository's existing FFmpeg runtime dependency for MP3 and Opus.

## Health Admission and Draining

`HealthState` uses atomics for warmup, draining, and stopped state plus a
`Notify` for active-transport changes. A successful synthesis admission returns
an RAII `AdmissionGuard`; dropping it decrements the active count exactly once,
including cancellation and panic paths.

Readiness is `warm && !draining && !stopped`. Liveness is `!stopped`. Health and
catalog calls do not consume synthesis admission. Unary requests hold admission
through response creation. Streaming and WebSocket requests hold it until the
body/socket and every owned stream context are cleaned up.

Shutdown sets draining, stops accepting synthesis, waits for active admission
or the grace deadline, closes protocol listeners, stops the engine, releases
native resources, and marks stopped.

## gRPC Implementation

`tonic-build` generates Rust messages and service traits from the existing
`styletts.proto`; generated Python files are neither copied nor translated.
The custom standard gRPC health service is implemented with the standard health
protobuf contract rather than hand-parsed bytes.

The unary method validates configuration, creates a stream, appends all text,
flushes, drains chunks, checks errors and sample-rate consistency, merges audio
and timing, and returns one response.

The server-streaming method creates one stream, appends and flushes text, then
maps each `AudioChunk` into a response. Client cancellation cancels the engine
stream through a drop guard.

Bidirectional incremental synthesis uses one `tokio::select!` loop over inbound
messages, engine audio, and cancellation. Its state enum is `AwaitingConfig`,
`Open`, or `Closed`. Config must be first and appears once. Text appends,
force-synthesis flushes, cancel increments stream generation, and inbound EOF
flushes remaining text before the response stream closes. No polling sleep is
used.

Catalog responses read immutable engine metadata. Unknown models map to
`NotFound`; invalid synthesis input maps to `InvalidArgument`; overload and
unready admission map to `Unavailable`; internal/native failures map to
`Internal` with structured logs but no native details on the wire.

## HTTP Route Implementation

Axum registers each route once:

```text
GET  /health
GET  /health/live
GET  /health/ready
GET  /livez
GET  /readyz
GET  /v1/models
GET  /v1/voices
POST /v1/text-to-speech/:voice_id
POST /v1/text-to-speech/:voice_id/with-timestamps
POST /v1/text-to-speech/:voice_id/stream
POST /v1/text-to-speech/:voice_id/stream/with-timestamps
GET  /v1/text-to-speech/:voice_id/stream-input
GET  /v1/text-to-speech/:voice_id/multi-stream-input
```

Serde request structs use `deny_unknown_fields` where the current contract
rejects unknown values. Custom deserializers distinguish missing, null, wrong
type, and out-of-range values so validation bodies retain the documented field
location, message, and type.

Query and body parsing resolve output format, model, voice, language,
normalization, seed, chunk schedule, and mapped voice settings before opening an
engine stream. Catalog validation therefore fails before WebSocket upgrade or
HTTP streaming headers.

Unary routes drain raw float chunks and encode once. Timing routes request
character alignment and format seconds. Plain routes request no alignment.
Streaming WAV is rejected before headers. Timing streams emit compact newline-
delimited JSON; plain streams emit codec bytes.

## WebSocket State Machines

The single-context handler uses these states:

```rust
enum SingleState { AwaitingInit, Open(StreamContext), Finalizing, Closed }
```

The first message must contain `text: " "`. It creates the engine stream and
freezes voice/generation settings. Open messages require text, enforce trailing
space, distinguish keepalive, append, conditional trigger, flush, and empty-text
finalization, and reject setting changes. Finalization drains generation, sends
one `isFinal`, closes the stream, then closes the socket.

The multi-context handler owns `HashMap<ContextId, StreamContext>` and one
bounded outbound writer queue. Messages create, update, flush, reinitialize, or
close a named context. Reinitialization drains and closes the old context before
installing the replacement. `close_socket` finalizes all contexts concurrently,
waits for queued responses, and then closes the socket.

Each `StreamContext` owns an engine stream, bounded generation notification
channel, audio-pump task, and resettable inactivity timer. Pump failure resolves
the finalization waiter with an error rather than deadlocking. Cleanup is
idempotent and always completes before its admission guard is dropped.

Outbound messages from multiple contexts pass through the one writer task so
frames never interleave at the socket implementation level. Each audio message
contains base64 audio, `isFinal: false`, optional original and normalized
alignment, and `contextId` for multi-context mode.

## Audio Implementation

The engine produces finite mono `f32` samples. Before integer conversion, Rust
clamps each sample to `[-1.0, 1.0]`. Output formats are a closed enum; permissive
string fallbacks are not retained.

- PCM uses little-endian signed 16-bit output.
- μ-law and A-law use direct Rust sample conversion tables.
- WAV writes a mono PCM16 RIFF header and frames at the requested rate.
- Resampling uses one stateful band-limited resampler per response.
- MP3 and Opus use one response-scoped FFmpeg process with piped PCM input and
  encoded output. The process is supervised, stderr is bounded, and a nonzero
  exit becomes an encoding error.

For compressed streaming, a background task feeds every PCM chunk into the same
encoder stdin while encoder stdout becomes the Axum response body. End of input
closes stdin, waits for the encoder, and then closes the body. This produces one
decodable stream rather than concatenated standalone files.

Encoding runs on bounded blocking capacity. Encoder startup failure occurs
before HTTP headers or WebSocket stream initialization. Mid-stream failure
terminates the body/socket according to its protocol contract and records the
typed cause.

## Alignment Formatting

Core alignment items contain text, source byte range, start time, and end time.
Model preprocessing retains mappings from normalized/preprocessed text back to
original UTF-8 ranges. Word and character conversion must reconstruct the
original input exactly for those alignment modes.

gRPC emits word, start milliseconds, and end milliseconds. WebSocket emits
characters, start milliseconds, and duration milliseconds. HTTP timing emits
characters plus start/end seconds. Chunk merging offsets both source ranges and
times; long-window merging additionally searches from the previous source end
so repeated text maps deterministically.

## Configuration

Typed YAML configuration separates server listeners, engine defaults, queue
capacities, automatic placement policy, and explicit placements. Unknown fields
are errors. Paths resolve relative to the configuration file.

Environment variables may select the configuration path and override listener
addresses or log level. Model placement, backend, platform, replica count,
slots, and batch size remain typed configuration rather than ad hoc environment
fallbacks.

The shape is explicit:

```yaml
server:
  grpc_address: "[::]:50051"
  http_address: "0.0.0.0:8000"
engine:
  stream_queue_capacity: 32
  device_queue_capacity: 256
  shutdown_grace_ms: 5000
  automatic_models:
    - model_id: stub
      artifact_dir: models/stub
      backend: onnx
      platform: cuda
      replicas: 2
      slots_per_replica: 2
      max_batch: 16
  placements:
    - model_id: stub-cpu
      artifact_dir: models/stub
      backend: onnx
      platform: cpu
      device: cpu
      replicas: 1
      slots_per_replica: 4
      max_batch: 8
```

Entries in `automatic_models` are resolved by the engine placement policy.
Entries in `placements` require an explicit device (`cpu` or `cuda:<index>`) and
are passed unchanged to core after validation.

The server refuses readiness when no model is loaded, a required listener fails,
an explicit placement is invalid, or warmup fails.

## Observability

Use `tracing` spans for protocol requests, streams, batches, native stages, and
shutdown. JSON fields preserve request ID, stream ID, model, voice, chunk,
replica, device, backend, platform, queue wait, batch size, inference duration,
and first-audio latency.

OpenTelemetry layers are configured in the server binary. Core modules depend
only on `tracing`. Native logging is forwarded through a callback that copies
messages immediately and never calls Rust while holding a TensorRT or scheduler
lock.

## Error Types and Wire Mapping

`TinferError` has concrete variants for configuration, validation, catalog,
overload, cancellation, unavailable placement, native stage, encoding,
invariant, and shutdown. Native-stage errors include model, stage, backend,
platform, device, and a sanitized cause.

HTTP validation uses the current `detail` shape. Catalog misses return 404,
unready/overloaded admission returns 503, and inference failures never return an
empty successful body. Errors after headers terminate the stream without
injecting JSON into audio bytes.

WebSocket validation sends the current error object and policy close code.
Inactivity sends an error associated with its context and closes that context or
socket as appropriate. gRPC mapping follows the status rules in the gRPC
section.

## Test Layout and Gates

Unit tests live beside focused modules. Cross-component tests live under:

```text
tinfer_rust/tests/
├── fixtures/
├── engine/
├── native/
├── grpc/
├── http/
├── websocket/
└── parity/
```

Contract fixtures are copied from the current Python tests without importing
Python at Rust test runtime. Temporary Python/Rust black-box parity tests may
start both servers during migration, but production Rust has no Python runtime
dependency.

Required gates are formatting, Clippy with warnings denied, unit/integration
tests, protobuf compatibility, CPU ONNX output, espeak spans, protocol golden
responses, and file/folder limit checks. GPU gates add ONNX CUDA, TensorRT,
backend parity, concurrent-slot stress, multi-GPU dispatch, stable primary-
context identity, warmup, recovery, and leak checks.
