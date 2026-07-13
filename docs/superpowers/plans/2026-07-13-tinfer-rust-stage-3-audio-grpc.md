# Tinfer Rust Stage 3 Audio and gRPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all current output formats and a Rust gRPC server covering health, catalog, unary, server-streaming, and bidirectional incremental synthesis.

**Architecture:** Engine audio remains mono `f32`; request-scoped encoders at the protocol boundary own resampler and compressed-stream state. Tonic handlers translate protobuf values to typed engine commands and never select devices or invoke native model code.

**Tech Stack:** Tonic, Prost, Tokio Stream, Tower, rubato, hound, in-process libmp3lame and libopus/Ogg bindings.

## Global Constraints

- Apply every constraint in `2026-07-13-tinfer-rust-port.md`.
- Copy the current protobuf source as the sole wire authority; do not edit field numbers or service names.
- Do not shell out to ffmpeg and do not make each audio chunk a separate MP3, Opus, or WAV file.
- Protocol handlers must release engine admission on success, error, cancellation, and disconnected clients.
- Run Cargo commands from `/workspace/tinfer/tinfer_rust` and Git command blocks from `/workspace/tinfer`.

---

### Task 1: PCM, Companding, WAV, and Resampling

**Files:**
- Create: `tinfer_rust/src/audio/{mod.rs,format.rs,resample.rs,pcm.rs,wav.rs}`
- Test: `tinfer_rust/tests/audio_lossless.rs`
- Test: `tinfer_rust/tests/fixtures/audio_golden.json`

**Interfaces:**
- Produces: `AudioFormat`, `AudioSpec`, and `AudioEncoder::{push(&[f32]) -> Result<Bytes>, finish() -> Result<Bytes>}`.

- [ ] **Step 1: Write failing golden tests**

```rust
#[test]
fn lossless_encodings_match_reference_bytes() {
    let samples = reference_wave();
    assert_eq!(encode_all(Pcm16, &samples), fixture("pcm_16000"));
    assert_eq!(encode_all(MuLaw, &samples), fixture("ulaw_8000"));
    assert_eq!(encode_all(ALaw, &samples), fixture("alaw_8000"));
    assert_eq!(decode_wav(encode_all(Wav, &samples)), quantized(samples));
}
```

Extract expected formats, sample rates, clipping, endianness, and rounding from current `audio_encoder.py` and `tmp_tests/tts_api` cases.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test audio_lossless`

Expected: compile failure for missing `AudioEncoder`.

- [ ] **Step 3: Implement typed format parsing and stateful encoding**

Parse only declared format strings. Resample once through a request-scoped rubato state, clamp explicitly, encode little-endian signed PCM16 and reference μ-law/A-law tables, and write exactly one WAV header in `finish`. Reject rate/channel combinations outside the current contract.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test audio_lossless`

Expected: byte fixtures and WAV decode tests pass.

```bash
git add tinfer_rust/src/audio tinfer_rust/tests/audio_lossless.rs tinfer_rust/tests/fixtures
git commit -m "feat(rust): add lossless audio encoders"
```

### Task 2: Streaming MP3 and Opus

**Files:**
- Create: `tinfer_rust/src/audio/{compressed.rs,mp3.rs,opus.rs}`
- Test: `tinfer_rust/tests/audio_compressed.rs`

**Interfaces:**
- Extends: `AudioEncoder` with response-scoped MP3 and Ogg Opus implementations.

- [ ] **Step 1: Write failing whole-stream tests**

```rust
#[test]
fn compressed_chunks_form_one_decodable_stream() {
    for format in [AudioFormat::Mp3, AudioFormat::Opus] {
        let bytes = encode_in_three_pushes(format, reference_wave());
        let decoded = decode_fixture_tool(format, &bytes);
        assert_duration_close(decoded.duration(), 1.0, 0.03);
        assert_eq!(count_container_starts(format, &bytes), 1);
    }
}
```

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test audio_compressed`

Expected: unsupported format error for MP3 and Opus.

- [ ] **Step 3: Implement in-process encoders**

Hold one libmp3lame encoder or one libopus encoder plus Ogg stream serial per response. `push` emits complete available frames/pages; `finish` drains delayed frames and writes one terminal page. Keep codec state out of `AudioChunk` and scheduler types.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test audio_compressed`

Expected: concatenated chunks decode as one stream and duration/tolerance checks pass.

```bash
git add tinfer_rust/src/audio tinfer_rust/tests/audio_compressed.rs
git commit -m "feat(rust): add stateful MP3 and Opus encoding"
```

### Task 3: Protobuf and gRPC Request Mapping

**Files:**
- Copy: `tinfer/tinfer/server/grpc/styletts.proto` to `tinfer_rust/proto/styletts.proto`
- Modify: `tinfer_rust/build.rs`
- Create: `tinfer_rust/src/server/{mod.rs,health.rs}`
- Create: `tinfer_rust/src/server/grpc/{mod.rs,convert.rs,error.rs}`
- Test: `tinfer_rust/tests/grpc_mapping.rs`

**Interfaces:**
- Produces: generated `pb` module, `TryFrom<pb::SynthesisRequest> for EngineSynthesis`, response conversions, and `GrpcErrorMap`.

- [ ] **Step 1: Write descriptor and mapping tests**

```rust
#[test]
fn checked_in_proto_descriptor_matches_reference() {
    assert_eq!(rust_descriptor_sha256(), reference_descriptor_sha256());
}

#[test]
fn invalid_alignment_maps_to_invalid_argument() {
    assert_eq!(map_request(request_with_alignment("bad")).unwrap_err().code(), Code::InvalidArgument);
}
```

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test grpc_mapping`

Expected: compile failure for missing generated `pb` module.

- [ ] **Step 3: Generate and map typed messages**

Use `tonic_build` from the copied proto. Convert every field explicitly, validate model/voice/language/output/alignment before admission, map catalog to `NotFound`, overload to `ResourceExhausted`, cancellation to `Cancelled`, validation to `InvalidArgument`, and inference to `Internal` with the current safe message contract.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test grpc_mapping`

Expected: descriptor and all mapping tables pass.

```bash
git add tinfer_rust/proto tinfer_rust/build.rs tinfer_rust/src/server tinfer_rust/tests/grpc_mapping.rs
git commit -m "feat(rust): generate and map gRPC contracts"
```

### Task 4: gRPC Services and Lifecycle

**Files:**
- Create: `tinfer_rust/src/server/grpc/{server.rs,catalog.rs,synthesis.rs,incremental.rs}`
- Modify: `tinfer_rust/src/server/health.rs`
- Test: `tinfer_rust/tests/grpc_integration.rs`

**Interfaces:**
- Produces: `GrpcServer::new(Arc<Engine>, Arc<HealthState>, GrpcConfig)`, `start`, `stop`, and `serve`.

- [ ] **Step 1: Write end-to-end service tests**

```rust
#[tokio::test]
async fn grpc_surface_matches_reference_contract() {
    let server = TestGrpcServer::start(stub_engine()).await;
    assert_health_and_catalog(&server).await;
    assert_unary_audio(&server).await;
    assert_server_stream_chunks(&server).await;
    assert_incremental_flush_and_final(&server).await;
}
```

Add concrete cases for invalid model/voice, client cancellation, midstream inference failure, readiness before/after warmup, drain rejection, and exactly-once admission release.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test grpc_integration`

Expected: compile failure for missing `GrpcServer`.

- [ ] **Step 3: Implement services over engine commands**

Unary collects one engine stream then encodes once. Server streaming creates one response-scoped encoder and forwards chunks in order. Bidirectional streaming runs separate inbound/outbound tasks around one engine stream; inbound text, conditional trigger, flush, and close become scheduler commands. On either task ending, cancel once and release health admission once.

Health is `Starting`, `Ready`, `Draining`, or `Stopped` plus active admission count. Readiness requires all configured placements warm. Stop rejects new admission, waits for active calls up to grace, then cancels remaining streams.

- [ ] **Step 4: Verify the stage and commit**

Run: `cargo test --test grpc_integration && cargo test --workspace --features onnx`

Expected: every gRPC mode and lifecycle case passes with the CPU stub.

```bash
git add tinfer_rust/src/server tinfer_rust/tests/grpc_integration.rs
git commit -m "feat(rust): serve complete gRPC API"
```
