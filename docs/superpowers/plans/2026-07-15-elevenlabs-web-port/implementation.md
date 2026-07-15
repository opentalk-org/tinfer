# ElevenLabs HTTP and WebSocket Rust Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the partial Rust web handlers with the complete ElevenLabs-compatible HTTP and WebSocket behavior defined by `tinfer/` and `tmp_tests/tts_api/`.

**Architecture:** `server/web` composes Axum and owns lifecycle, `server/http` owns typed HTTP contracts and four synthesis modes, and `server/websocket` owns single- and multi-context state machines. An object-safe service adapter lets transport tests control engine output while production delegates to `AsyncEngine` and `AsyncStream`.

**Tech Stack:** Rust 2024, Axum 0.8, Tokio 1.48, Serde, futures-util, tokio-stream, Tower, reqwest, and tokio-tungstenite.

## Global Constraints

- `tinfer/tinfer/server/websocket/` and `tmp_tests/tts_api/` are contract authority; official ElevenLabs TTS documentation only verifies concepts.
- Production Rust has no Python dependency. Leave gRPC, engine, native model, and unrelated StyleTTS2/CUDA behavior unchanged.
- Validate query, body, format, model, voice, and language before stream creation; validate WebSockets before upgrade.
- Use bounded channels and exactly one owner for closing each engine stream.
- Keep every file below 300 lines and every folder below 16 files.
- Run Cargo commands from `/workspace/tinfer/tinfer_rust` and Git commands from `/workspace/tinfer`.

## File Structure

```text
tinfer_rust/src/server/
├── http/{mod,error,query,schema,catalog,response,synthesis,streaming}.rs
├── web/{mod,server,service}.rs
└── websocket/{mod,schema,response,writer,context,single,multi}.rs
tinfer_rust/tests/
├── web_integration.rs
└── web_integration/{support,contracts,http,streaming,single,multi,lifecycle}.rs
```

Delete `src/server/web/wire.rs` after moving its types. `web_integration.rs` declares the seven test modules so Cargo integration-test discovery and top-level folder count stay stable.

---

### Task 1: Typed Contracts and Test Harness

**Files:** Create `src/server/http/{mod.rs,error.rs,query.rs,schema.rs}` and `tests/web_integration/{support.rs,contracts.rs}`; modify `src/server/{mod.rs,web/mod.rs}` and `tests/web_integration.rs`.

**Interfaces:** Produce `Transport`, `TextNormalization`, `SpeechQuery`, `SpeechRequest`, `VoiceSettings`, `GenerationConfig`, `ApiError`, `parse_query`, `parse_speech`, `to_stream_params`, and loopback `TestServer::{start,url,post}`; consume `AudioFormat`, `AlignmentType`, `StreamParams`, and model metadata.

- [ ] **Step 1: Write failing contract tests.** Replace `tests/web_integration.rs` with `mod web_integration { mod support; mod contracts; }`. In `contracts.rs`, use this complete format table and assert Python's missing-text body:

```rust
const VALID_FORMATS: &[&str] = &[
 "mp3_22050_32","mp3_24000_48","mp3_44100_32","mp3_44100_64","mp3_44100_96","mp3_44100_128","mp3_44100_192",
 "pcm_8000","pcm_16000","pcm_22050","pcm_24000","pcm_32000","pcm_44100","pcm_48000",
 "wav_8000","wav_16000","wav_22050","wav_24000","wav_32000","wav_44100","wav_48000",
 "ulaw_8000","alaw_8000","opus_48000_32","opus_48000_64","opus_48000_96","opus_48000_128","opus_48000_192",
];
assert_eq!(post_json("/v1/text-to-speech/default", json!({})).await,
 json!({"detail":[{"loc":["body","text"],"msg":"Field required","type":"missing"}]}));
```

Add table rows from `test_contract_types.py` for query defaults/fields, HTTP-only WebSocket formats, unknown fields, normalization, booleans, seed/timeout/latency bounds, voice bounds, schedules, dictionary locators, neighboring text/request IDs, optional nulls, wrong types, and accepted no-effect fields.

- [ ] **Step 2: Verify RED.** Run `cargo test --test web_integration contracts -- --nocapture`; expect missing-text/default/field assertions to fail against the partial handler.

- [ ] **Step 3: Implement strict parsing** with these exact construction interfaces:

```rust
pub(crate) enum Transport { Http, WebSocket }
pub(crate) enum TextNormalization { Auto, On, Off }
pub(crate) struct SpeechQuery { pub model_id: Option<String>, pub output_format: AudioFormat, pub language_code: Option<String>, pub sync_alignment: bool, pub inactivity_timeout: Duration, pub auto_mode: bool, pub seed: Option<u32>, pub normalization: TextNormalization }
pub(crate) struct VoiceSettings { pub speed: Option<f64>, pub alpha: Option<f64>, pub beta: Option<f64>, pub stability: Option<f64>, pub similarity_boost: Option<f64>, pub style: Option<f64>, pub use_speaker_boost: Option<bool> }
pub(crate) struct GenerationConfig { pub chunk_length_schedule: Vec<usize> }
pub(crate) struct SpeechRequest { pub text: String, pub model_id: Option<String>, pub language_code: Option<String>, pub voice_settings: VoiceSettings, pub generation_config: GenerationConfig, pub seed: Option<u32>, pub normalization: Option<TextNormalization> }
pub(crate) fn parse_query(values: &HashMap<String,String>, transport: Transport) -> Result<SpeechQuery,ApiError>;
pub(crate) fn parse_speech(value: Value) -> Result<SpeechRequest,ApiError>;
pub(crate) fn to_stream_params(query: &SpeechQuery, speech: &SpeechRequest, alignment: AlignmentType) -> StreamParams;
```

Use closed field sets and typed helpers for optional strings, booleans, bounded integers/floats, arrays, and nested objects. `ApiError::Issue` renders the single-item `detail` array; other validation failures use status 422.

- [ ] **Step 4: Verify GREEN and commit.** Run `cargo test --test web_integration contracts -- --nocapture`; expect the parser matrix on the registered HTTP and single-socket routes to pass. Commit with `git add tinfer_rust/src/server tinfer_rust/tests/web_integration* && git commit -m "feat(rust): define ElevenLabs web contracts"`.

### Task 2: Service Adapter, Catalog, and Unary HTTP

**Files:** Create `src/server/web/service.rs`, `src/server/http/{catalog.rs,response.rs,synthesis.rs}`, and `tests/web_integration/http.rs`; modify `src/server/{http/mod.rs,web/mod.rs}` and `tests/web_integration.rs` to declare `mod http;`.

**Interfaces:** Produce object-safe `SpeechEngine`, `SpeechStream`, `DynEngine`, `DynStream`, `AppState`, `http::routes() -> Router<AppState>`, catalog handlers, unary formatters, and controlled `TestServer::{controlled,release_generation,stream_closes}`.

- [ ] **Step 1: Write failing tests** for model-array/voice-envelope identity, default and unknown model, unknown voice, every MIME family, empty speech, language mapping, draining 503, inference 500, seconds timing, normalized alignment, and admission returning to zero:

```rust
let audio = server.post("/v1/text-to-speech/default?output_format=pcm_24000", json!({"text":"AB"})).await;
assert_eq!(audio.status(), 200); assert!(!audio.bytes().await.unwrap().is_empty());
let timed = server.post("/v1/text-to-speech/default/with-timestamps?output_format=pcm_24000", json!({"text":"AB"})).await.json::<Value>().await.unwrap();
assert_eq!(timed["alignment"]["characters"], json!(["A","B"]));
```

- [ ] **Step 2: Verify RED.** Run `cargo test --test web_integration http -- --nocapture`; expect strict errors and timing/parameter assertions to fail.

- [ ] **Step 3: Implement the adapter and handlers** using these object-safe methods:

```rust
trait SpeechEngine: Send+Sync { fn model_infos(&self)->BoxFuture<'_,Result<Vec<ModelInfo>>>; fn voices<'a>(&'a self,model:&'a str)->BoxFuture<'a,Result<Vec<String>>>; fn create<'a>(&'a self,model:&'a str,voice:&'a str,params:StreamParams)->BoxFuture<'a,Result<DynStream>>; }
trait SpeechStream: Send+Sync { fn add<'a>(&'a self,text:&'a str)->BoxFuture<'a,Result<()>>; fn try_generate(&self)->BoxFuture<'_,Result<()>>; fn flush(&self)->BoxFuture<'_,Result<()>>; fn finish(&self)->BoxFuture<'_,Result<()>>; fn receive(&self)->BoxFuture<'_,Result<(Option<AudioChunk>,bool)>>; fn cancel(&self)->BoxFuture<'_,Result<()>>; fn close(&self)->BoxFuture<'_,Result<()>>; }
```

Resolve catalog and effective language before `create`. Unary calls add, finish, drains marked output, merges chunks, encodes once, closes once, and holds admission through response construction.

- [ ] **Step 4: Verify GREEN and commit.** Run `cargo test --test web_integration -- --nocapture`; expect contract and HTTP tests to pass with exact envelopes, status bodies, MIME types, timing arrays, and no leaked admission. Commit `feat(rust): port catalog and unary HTTP synthesis` with the task files.

### Task 3: Genuine HTTP Streaming

**Files:** Create `src/server/http/streaming.rs` and `tests/web_integration/streaming.rs`; modify `src/server/http/mod.rs` and `tests/web_integration.rs` to declare `mod streaming;`.

**Interfaces:** Produce streaming audio and newline-delimited timestamp handlers backed by bounded `mpsc`; consume Task 2 resolution, ownership, formatting, and `AudioEncoder`.

- [ ] **Step 1: Write failing tests** for first frame before controlled release, ordered timing records, `text/event-stream`, WAV rejection before headers, one decodable MP3/Opus stream, disconnect cancellation, mid-stream failure, and close/admission counts:

```rust
let mut body = server.post_stream("/v1/text-to-speech/default/stream?output_format=pcm_24000", json!({"text":"AB"})).await;
assert_eq!(body.chunk().await.unwrap().unwrap().len(), 4);
assert_eq!(server.health.active_admissions(), 1); server.release_generation();
while body.chunk().await.unwrap().is_some() {} assert_eq!(server.health.active_admissions(), 0);
```

- [ ] **Step 2: Verify RED.** Run `cargo test --test web_integration streaming -- --nocapture`; expect failure because stream routes alias unary responses.

- [ ] **Step 3: Implement bounded pumps.** Validate and create the encoder before headers. A task owns `Admission`, `DynStream`, and `AudioEncoder`; it sends encoded chunks or compact timed JSON plus `b'\n'`, sends the encoder tail, then closes. Receiver closure cancels then closes; engine/codec errors terminate without JSON in media bytes.

- [ ] **Step 4: Verify GREEN and commit.** Re-run Task 3 tests; expect observable early bytes, ordered records, and close counts of one. Commit `feat(rust): stream HTTP audio and timing` with Task 3 files.

### Task 4: Single-Context WebSocket

**Files:** Create `src/server/websocket/{mod.rs,schema.rs,response.rs,writer.rs,context.rs,single.rs}` and `tests/web_integration/single.rs`; modify `src/server/web/mod.rs` and `tests/web_integration.rs` to declare `mod single;`.

**Interfaces:** Produce `websocket::routes() -> Router<AppState>`, `SingleSession`, `StreamContext`, `ClientCommand`, `ServerEvent`, and one bounded `SocketWriter`.

- [ ] **Step 1: Write failing tests** for valid init/audio/final below plus invalid init, binary/non-object/malformed messages, trailing space, keepalive, conditional trigger, flush, rejected client `isFinal`, unchanged/changed settings, alignment, handshake errors, inactivity, pump failure, disconnect-before-release, close codes, and admission cleanup:

```rust
ws.send(Message::text(r#"{"text":" ","generation_config":{"chunk_length_schedule":[80]}}"#)).await.unwrap();
ws.send(Message::text(r#"{"text":"Hello ","flush":true}"#)).await.unwrap();
assert_eq!(receive_json(&mut ws).await["isFinal"], false);
ws.send(Message::text(r#"{"text":""}"#)).await.unwrap();
assert_eq!(receive_until_final(&mut ws).await, json!({"isFinal":true}));
```

- [ ] **Step 2: Verify RED.** Run `cargo test --test web_integration single -- --nocapture`; expect failures on settings, errors, alignment, inactivity, writer ordering, and close semantics.

- [ ] **Step 3: Implement explicit state and event types:**

```rust
enum SingleState { AwaitingInitialization, Active(StreamContext), Finalizing, Closed }
enum ClientCommand { Initialize(Initialization), Append(String), TryGenerate, Flush, KeepAlive, Finalize }
enum ServerEvent { Audio { audio: String, alignment: Option<WsAlignment>, normalized_alignment: Option<WsAlignment>, context_id: Option<String> }, Final { context_id: Option<String> }, Error { error: String, context_id: Option<String> } }
```

The reader validates commands, `SingleSession` changes state, `StreamContext` owns stream/pump/timer, and `SocketWriter` alone sends. Finalize drains audio, queues one final, closes stream, drains writer, and closes 1000; policy/internal errors queue an error before 1008/1011.

- [ ] **Step 4: Verify GREEN and commit.** Re-run Task 4 tests; expect all transitions, fields, codes, timeouts, and counts to pass. Commit `feat(rust): port single-context WebSocket synthesis` with Task 4 files.

### Task 5: Multi-Context WebSocket

**Files:** Create `src/server/websocket/multi.rs` and `tests/web_integration/multi.rs`; modify `src/server/websocket/{mod.rs,schema.rs,response.rs,context.rs}` and `tests/web_integration.rs` to declare `mod multi;`.

**Interfaces:** Produce `/multi-stream-input` and `MultiSession { contexts: HashMap<ContextId, StreamContext> }`; consume Task 4 writer/events/contexts.

- [ ] **Step 1: Write failing tests** for default/named creation, initialization, invalid context IDs, textless flush, keepalive, close/reuse, settings reinitialization, schedules/alignment, per-context inactivity/failure, malformed frames, disconnect cleanup, final ordering, and serialized output:

```rust
send_json(&mut ws,json!({"context_id":"a","text":" "})).await;
send_json(&mut ws,json!({"context_id":"a","text":"Alpha ","flush":true})).await;
send_json(&mut ws,json!({"context_id":"b","text":"Beta ","flush":true})).await;
assert_eq!(receive_context_ids(&mut ws,4).await,["a","a","b","b"]);
send_json(&mut ws,json!({"close_socket":true})).await;
assert_eq!(receive_final_ids(&mut ws,2).await,HashSet::from(["a","b"]));
```

- [ ] **Step 2: Verify RED.** Run `cargo test --test web_integration multi -- --nocapture`; expect 404 because the route is absent.

- [ ] **Step 3: Implement context ownership.** Parse create/update/flush/close-context/close-socket commands. Reinitialization awaits prior final and closure. Timeout removes only the matching generation. Socket closure concurrently finalizes the map, waits for writer acknowledgements, then closes 1000.

- [ ] **Step 4: Verify GREEN and commit.** Re-run Task 5 tests; expect isolated text/settings, required `contextId`, exact finals, and no active streams/admissions. Commit `feat(rust): port multi-context WebSocket synthesis` with Task 5 files.

### Task 6: Lifecycle and Full Verification

**Files:** Modify `src/server/web/{mod.rs,server.rs}`, `tests/web_integration.rs`, and all `tests/web_integration/*.rs`; create `tests/web_integration/lifecycle.rs` and declare `mod lifecycle;`; delete `src/server/web/wire.rs`.

**Interfaces:** Produce final `WebServer::{new,from_engine,start,stop}` and the permanent parity suite; consume all protocol modules and `HealthState::{set,drained}`.

- [ ] **Step 1: Write failing tests** for health bodies, repeated-start assertion, stop-before-start, draining rejection, grace timeout, HTTP/socket disconnects, and admission release after cleanup:

```rust
let socket=server.open_initialized_socket().await; let stop=tokio::spawn(server.stop());
assert_eq!(server.health.state(),ServingState::Draining);
assert_eq!(server.post_speech("blocked").await.status(),503);
drop(socket); stop.await.unwrap().unwrap();
assert_eq!(server.health.state(),ServingState::Stopped);
```

- [ ] **Step 2: Verify RED.** Run `cargo test --test web_integration lifecycle -- --nocapture`; expect listener completion or health ordering assertions to fail.

- [ ] **Step 3: Complete lifecycle.** Compose routes once; mark draining before shutdown; await listener and `health.drained()` inside grace; abort only at deadline; then mark stopped. Compare Rust test names with all 44 `tmp_tests/tts_api/test_*.py` tests and place every missing assertion in its focused module before deleting `wire.rs`.

- [ ] **Step 4: Run gates:**

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --test web_integration -- --nocapture
cargo test
find src tests -type f -not -path '*/target/*' -exec sh -c 'n=$(wc -l < "$1"); test "$n" -lt 300 || { echo "$1: $n"; exit 1; }' _ {} \;
find src tests -type d -not -path '*/target/*' -exec sh -c 'n=$(find "$1" -mindepth 1 -maxdepth 1 -type f | wc -l); test "$n" -lt 16 || { echo "$1: $n files"; exit 1; }' _ {} \;
```

Expected: format unchanged, no Clippy warnings, all tests pass, and limit checks print nothing.

- [ ] **Step 5: Commit.** Run `git add tinfer_rust/src/server tinfer_rust/tests/web_integration* && git commit -m "feat(rust): complete ElevenLabs web protocol port"`.
