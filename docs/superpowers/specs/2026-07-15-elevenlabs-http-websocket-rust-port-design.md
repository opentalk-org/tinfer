# ElevenLabs HTTP and WebSocket Rust Port Design

## Goal

Port the complete ElevenLabs-compatible HTTP and WebSocket surface implemented
in `tinfer/tinfer/server/websocket/` into `tinfer_rust`. The Python service and
`tmp_tests/tts_api/` define compatibility behavior. Official ElevenLabs TTS API
documentation is a verification reference, not a reason to add endpoints that
the Python service does not implement.

The production Rust server must not depend on Python. The port must leave the
gRPC API, engine behavior, native models, and unrelated StyleTTS2/CUDA work
unchanged.

## Supported Surface

The Rust router exposes each route exactly once:

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

Request fields, query parameters, defaults, numeric bounds, output formats,
catalog responses, response casing, alignment shapes, MIME types, and accepted
no-effect compatibility fields match the Python parsers and formatters.

## Architecture

The minimal `server/web` implementation is replaced with focused
modules while preserving `WebServer` as the public lifecycle facade:

- `server/web` owns router composition, shared state, and listener lifecycle.
- `server/http` owns strict contracts, catalog handlers, HTTP errors, unary
  synthesis, streaming synthesis, and response formatting.
- `server/websocket` owns wire commands/events, shared stream contexts, the
  single-context state machine, the multi-context state machine, and the sole
  socket writer for each connection.

HTTP and WebSocket code depend only on typed engine, audio, and health APIs.
Protocol values are translated into `StreamParams` before entering the engine;
HTTP-specific strings and raw JSON do not enter scheduler or model code.

All channels introduced by the port are bounded. Each active transport owns an
admission guard. Each active synthesis context owns exactly one engine stream,
and ownership determines which task is responsible for closing it.

## Typed Contracts and Validation

Serde-backed structs and enums represent query parameters, request bodies,
voice settings, generation settings, client commands, server events, context
IDs, and output formats. Unknown fields and unsupported enum values are errors.
Custom validation preserves distinctions that affect the Python wire contract,
including missing fields, nulls, wrong JSON types, empty strings, and values
outside documented bounds.

Validation order is deterministic. Query, output format, model, voice, and
language validation completes before engine stream creation. WebSocket query
and catalog validation completes before protocol upgrade.

The loaded default model is used when `model_id` is omitted. Unsupported
languages resolve to the model's default language, matching the Python service.
Accepted compatibility fields that have no engine effect are validated and
then discarded explicitly.

## HTTP Data Flow

Unary endpoints acquire admission, parse the request, create one engine stream,
append text, finish generation, drain all chunks, reject inference errors, and
merge audio and alignment. Plain responses encode one audio body. Timestamp
responses base64-encode the audio and emit original and normalized alignment.
Admission remains held through response construction.

Streaming endpoints perform the same validation before response headers. They
retain admission, the engine stream, and one response-scoped encoder until the
body completes or the client disconnects. Plain streaming emits codec bytes in
chunk order. Timestamp streaming emits compact newline-delimited JSON records.
WAV streaming is rejected before headers, matching Python behavior. Compressed
streams use one stateful encoder so the response is one decodable media stream.

Disconnect, cancellation, encoding failure, or inference failure terminates the
stream and closes its engine stream exactly once. Failures after headers never
inject JSON into audio bytes.

## Single-Context WebSocket

The single-context session has explicit states: awaiting initialization,
active, finalizing, and closed. The first client message requires `text: " "`
and freezes voice and generation settings. Active messages support append,
conditional generation, flush, keepalive, and empty-text finalization. Speech
text must retain the Python trailing-space rule, and settings cannot change
after initialization.

One bounded writer serializes all outbound events. Audio events contain base64
audio, `isFinal: false`, and alignment fields when requested. Successful
finalization drains generated audio, emits exactly one `isFinal` event, closes
the engine stream, and closes the socket normally.

The inactivity timer resets only for contract-defined activity. Expiration
closes the context and socket. Client disconnect and task cancellation follow
the same ownership cleanup path without duplicate stream closure.

## Multi-Context WebSocket

The multi-context session owns a map from validated context IDs to independent
stream contexts. The initial message creates the default or named context.
Later messages create, append, flush, keep alive, close, reuse, or reinitialize
contexts according to the Python state transitions.

Reinitializing settings finalizes the prior context before creating its
replacement. Each response carries `contextId`. A single bounded writer task
serializes events from all contexts so frames cannot overlap. Context-specific
inactivity removes and closes only that context while the socket and other
contexts remain active.

`close_socket` finalizes all active contexts, waits until their queued audio and
final events have been written, and then closes the socket. Disconnect and
server shutdown close every owned context exactly once.

## Errors and Lifecycle

HTTP body validation preserves the Python `422 {"detail":[...]}` issue shape.
Unknown models and voices return 404, unavailable admission returns 503, and
synthesis or encoding failures before headers return 500. Health and catalog
requests do not consume synthesis admission.

Malformed WebSocket frames and invalid state transitions send an `error` event
and close with code 1008. Unexpected server failures close with 1011. Successful
completion closes with 1000. Handshake-time validation errors remain ordinary
HTTP responses.

`WebServer::stop` marks health as draining, stops further admission, requests Axum
shutdown, and waits for active transports up to the configured grace period.
The process lifecycle remains responsible for stopping the engine.

## Testing

Every behavior in `tmp_tests/tts_api/` is represented in permanent Rust tests:

- query/body parsing, defaults, bounds, unknown fields, and no-effect fields;
- output formats, resampling, response headers, and alignment casing;
- model and voice discovery plus default and missing catalog entries;
- unary audio, unary timestamps, streaming audio, and streaming timestamps;
- admission, draining, inference failures, disconnects, and exact cleanup;
- single-context initialization, messages, triggers, settings, timeout, errors,
  alignment, finalization, and close codes;
- multi-context isolation, ordering, creation, reuse, reinitialization, timeout,
  context closure, socket closure, and disconnect cleanup.

Focused parser and state-machine unit tests supplement loopback Axum HTTP and
WebSocket integration tests using the Rust stub engine. Python is not imported
or launched by the permanent Rust suite.

Completion requires `cargo fmt --check`, Clippy with warnings denied, targeted
protocol tests, the full applicable Rust test suite, and checks that every file
is below 300 lines and every folder is below 16 files.

## External Verification References

- <https://elevenlabs.io/docs/api-reference/text-to-speech/convert>
- <https://elevenlabs.io/docs/api-reference/text-to-speech/stream>
- <https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps>
- <https://elevenlabs.io/docs/api-reference/text-to-speech/stream-with-timestamps>

These references verify the intended ElevenLabs-compatible concepts. When they
differ from the scoped Python implementation, the Python contract controls this
port.
