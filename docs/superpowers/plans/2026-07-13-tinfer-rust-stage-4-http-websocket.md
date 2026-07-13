# Tinfer Rust Stage 4 HTTP and WebSocket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the current ElevenLabs-compatible discovery, HTTP synthesis, single-context WebSocket, and multi-context WebSocket surface entirely in Rust.

**Architecture:** Axum routes map JSON/query values into the same typed engine requests as gRPC. Each WebSocket connection owns a Rust state machine and bounded writer queue; each active context owns one engine stream. Protocol state never enters the scheduler and model/native code never sees HTTP values.

**Tech Stack:** Axum, Tower, Tokio, Serde, serde_json, futures, tokio-tungstenite integration through Axum.

## Global Constraints

- Apply every constraint in `2026-07-13-tinfer-rust-port.md`.
- Treat current `tmp_tests/tts_api/` tests and the route table in `tinfer/tinfer/server/websocket/server.py` as contract authority.
- Preserve current status codes, JSON shapes, close codes, error timing, final messages, alignment offsets, and content types.
- Use bounded inbound, engine-output, and socket-writer channels; overload is explicit.
- Run Cargo commands from `/workspace/tinfer/tinfer_rust` and Git command blocks from `/workspace/tinfer`.

---

### Task 1: Route, Query, and JSON Contracts

**Files:**
- Create: `tinfer_rust/src/server/http/{mod.rs,routes.rs,query.rs,schema.rs,error.rs}`
- Create: `tinfer_rust/src/server/websocket/{mod.rs,schema.rs}`
- Test: `tinfer_rust/tests/http_contracts.rs`

**Interfaces:**
- Produces: `HttpSynthesis`, `SpeechRequest`, `SpeechQuery`, `VoiceSettings`, `OutputFormat`, `ApiError`, and `build_router(HttpState) -> Router`.

- [ ] **Step 1: Write failing route and parser tests**

```rust
#[test]
fn router_contains_every_compatibility_route() {
    assert_routes([
        "/health", "/health/live", "/health/ready", "/v1/models", "/v1/voices",
        "/v1/text-to-speech/:voice_id", "/v1/text-to-speech/:voice_id/stream",
        "/v1/text-to-speech/:voice_id/with-timestamps",
        "/v1/text-to-speech/:voice_id/stream/with-timestamps",
        "/v1/text-to-speech/:voice_id/stream-input",
        "/v1/text-to-speech/:voice_id/multi-stream-input",
    ]);
}
```

Add table cases copied from `query_parser.py` and `speech_parser.py` for every field, default, numeric bound, unknown/invalid value, language, model, voice setting, chunk schedule, and output format.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test http_contracts`

Expected: compile failure for missing `build_router` and schemas.

- [ ] **Step 3: Implement strict typed parsing**

Use Serde request structs and enums, but collect validation issues in the exact current order before constructing `EngineSynthesis`. Reject malformed structured fields instead of accepting raw dictionaries. Build the complete router with stub handlers so route existence can pass independently.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test http_contracts`

Expected: route and parser matrices pass.

```bash
git add tinfer_rust/src/server/http tinfer_rust/src/server/websocket tinfer_rust/tests/http_contracts.rs
git commit -m "feat(rust): define HTTP and WebSocket contracts"
```

### Task 2: Discovery and Four HTTP Synthesis Modes

**Files:**
- Create: `tinfer_rust/src/server/http/{catalog.rs,synthesis.rs,response.rs}`
- Test: `tinfer_rust/tests/http_integration.rs`

**Interfaces:**
- Consumes: engine streams, health admission, `AudioEncoder`, and Task 1 contracts.
- Produces: health/catalog handlers and unary/streaming audio/timestamp handlers.

- [ ] **Step 1: Write end-to-end HTTP tests**

```rust
#[tokio::test]
async fn all_http_modes_match_contract() {
    let app = test_app(stub_engine());
    assert_discovery(&app).await;
    assert_audio_route(&app, false).await;
    assert_audio_route(&app, true).await;
    assert_timestamp_route(&app, false).await;
    assert_timestamp_route(&app, true).await;
}
```

Port concrete `tmp_tests/tts_api` assertions for headers, model/voice JSON, formats, timing arrays, source spans, validation bodies, missing resources, inference failure before/after headers, disconnect, and admission cleanup.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test http_integration`

Expected: stub handlers return not-implemented responses.

- [ ] **Step 3: Implement route handlers**

Unary routes flush one engine stream, merge audio/alignment in Rust, and emit one body. Streaming routes keep one encoder, emit in chunk order, and finish the container once. Timestamp routes format character arrays from structured alignment only at the response boundary. Use one drop guard for cancellation plus admission release.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test http_integration`

Expected: all four synthesis forms, discovery, health, errors, and cleanup pass.

```bash
git add tinfer_rust/src/server/http tinfer_rust/tests/http_integration.rs
git commit -m "feat(rust): serve discovery and HTTP synthesis"
```

### Task 3: Single-Context WebSocket State Machine

**Files:**
- Create: `tinfer_rust/src/server/websocket/{single.rs,session.rs,writer.rs,response.rs}`
- Test: `tinfer_rust/tests/websocket_single.rs`

**Interfaces:**
- Produces: `SingleSession::accept(SpeechQuery, EngineStream)`, `handle(ClientCommand)`, and typed `ServerEvent` values.

- [ ] **Step 1: Write state-transition tests**

```rust
#[tokio::test]
async fn single_socket_enforces_initialization_flush_and_final() {
    let socket = TestSocket::connect(test_app(stub_engine())).await;
    socket.send(init_message()).await;
    socket.send(text_message("hello", false)).await;
    socket.send(flush_message()).await;
    assert_audio_then_exactly_one_final(socket).await;
}
```

Add initialization-required, duplicate/changed settings, empty keepalive, conditional trigger, inactivity, malformed JSON, invalid query, client close, server drain, inference error, and writer-backpressure cases from current tests.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test websocket_single`

Expected: WebSocket upgrade succeeds but no session behavior exists.

- [ ] **Step 3: Implement explicit states and tasks**

```rust
enum SingleState { AwaitingInitialization, Active(ActiveContext), Finalizing, Closed }
enum ClientCommand { Initialize(Init), Append(Text), ConditionalTrigger, Flush, KeepAlive, Close }
```

One reader validates frames into commands, one state-machine task owns transitions and engine commands, and one writer serializes bounded `ServerEvent`s. Reset inactivity only on contract-defined activity. Flush drains admitted output, emits final once, closes the stream, then closes the socket.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test websocket_single`

Expected: every transition, close code, error message, finalization, and cleanup case passes.

```bash
git add tinfer_rust/src/server/websocket tinfer_rust/tests/websocket_single.rs
git commit -m "feat(rust): add single-context WebSocket synthesis"
```

### Task 4: Multi-Context WebSocket and Server Lifecycle

**Files:**
- Create: `tinfer_rust/src/server/websocket/{multi.rs,context.rs}`
- Create: `tinfer_rust/src/server/http/server.rs`
- Test: `tinfer_rust/tests/websocket_multi.rs`
- Test: `tinfer_rust/tests/http_server_lifecycle.rs`

**Interfaces:**
- Produces: `MultiSession` with typed context IDs and `HttpServer::{start,stop,serve}` sharing `HealthState` and `Engine`.

- [ ] **Step 1: Write interleaving and lifecycle tests**

```rust
#[tokio::test]
async fn contexts_interleave_without_crossing_state() {
    let socket = TestSocket::connect_multi(test_app(stub_engine())).await;
    send_interleaved_contexts(&socket, ["a", "b"]).await;
    assert_context_outputs_are_ordered_and_isolated(socket, ["a", "b"]).await;
}
```

Cover context create/update/reinitialize/finalize, duplicate IDs, per-context inactivity, socket finalize, one failed context, exactly-once finals, max contexts, disconnect cleanup, readiness, draining, and repeated start/stop.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test websocket_multi --test http_server_lifecycle`

Expected: multi route has no state machine and lifecycle type is missing.

- [ ] **Step 3: Implement context ownership and graceful stop**

One `MultiSession` owns `HashMap<ContextId, ContextState>` and the sole socket writer. Each context owns one engine stream and inactivity key; completions include context ID and generation so reinitialized contexts discard stale output. Socket finalization flushes active contexts, drains their outputs, emits one final per context and one socket final according to the current contract.

`HttpServer::stop` marks draining, stops admission, waits for active transports up to grace, sends session shutdown, stops Axum accept, and leaves engine shutdown to the process lifecycle owner.

- [ ] **Step 4: Verify the stage and commit**

Run: `cargo test --test http_contracts --test http_integration --test websocket_single --test websocket_multi --test http_server_lifecycle`

Expected: the full Rust HTTP/WSS contract suite passes.

```bash
git add tinfer_rust/src/server tinfer_rust/tests
git commit -m "feat(rust): complete multi-context WebSocket server"
```
