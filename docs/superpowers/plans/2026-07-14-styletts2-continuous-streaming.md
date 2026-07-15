# StyleTTS2 Continuous Streaming Implementation Plan

> **For agentic workers:** Execute inline task-by-task. Do not dispatch subagents and do not commit.

**Goal:** Stream arbitrarily long StyleTTS2 audio continuously while retaining opportunistic batching and low first-audio latency.

**Architecture:** The generic engine separately schedules model starts and acoustic continuations until the model marks a text chunk complete, pinning all continuation work to the same loaded model entry. StyleTTS2 owns token splitting, duration expansion, fixed 128-frame acoustic windows, and persistent per-stream state. Existing models remain single-call by returning `complete: true`.

**Tech Stack:** Rust, crossbeam channels, CXX, C++20, ONNX Runtime, TensorRT, CUDA.

## Global Constraints

- Do not commit.
- Do not use subagents.
- Keep model-specific behavior below `src/models/styletts2/`.
- Keep A dynamic up to 512 model tokens.
- Run B and C with 128 acoustic frames: 64 pre-context, 48 emitted core, 16 post-context.
- Preserve one loaded model and CUDA context per config entry; continuations stay on that entry.
- Prefer focused verification over repeated full-suite runs.
- Do not add fixed text slots or conversation-audio conditioning; those require the future fine-tuned model described in `docs/superpowers/specs/2026-07-14-styletts2-conversation-conditioning.md`.

---

### Task 1: Generic partial model outputs

**Files:**
- Modify: `tinfer_rust/src/types.rs`
- Modify: `tinfer_rust/src/models/base/model.rs`
- Modify: `tinfer_rust/src/models/stub/mod.rs`
- Modify: `tinfer_rust/src/engine/scheduler.rs`
- Modify: `tinfer_rust/src/engine/registry.rs`
- Modify: `tinfer_rust/src/engine/engine.rs`
- Test: `tinfer_rust/tests/engine.rs`

**Interfaces:**
- Add `stream_id: u64` and `operation: ModelOperation` to `ModelRequest`.
- Add `complete: bool` to `ModelOutput`.
- Add `Model::close_stream(stream_id)` for releasing model-owned stream state.
- Keep an active text span in the scheduler until `complete` is true.

- [ ] Add an engine test model behavior where one text chunk returns two audio outputs before completion.
- [ ] Verify the stream receives both outputs in order and the next text chunk is not started early.
- [ ] Pin the continuation to its original `Registry::EntryId`; never migrate state between configured model entries.
- [ ] Prioritize continuations whose produced-audio duration trails wall time.
- [ ] Call `close_stream` on close, cancel, unload, shutdown, and terminal model failure.
- [ ] Preserve current behavior by making the stub return `complete: true`.
- [ ] Run `cargo test --test engine`.

The scheduler state should be explicit:

```rust
struct ActiveChunk {
    entry: EntryId,
    span: Range<usize>,
    text: String,
}
```

`ModelOperation::Start` runs A and returns an incomplete result with no audio. `ModelOperation::Continue` runs exactly one B+C window. An incomplete acoustic result sends its audio immediately and leaves `ActiveChunk` ready for another continuation. A complete result clears it and allows the next prepared text chunk.

### Task 2: StyleTTS2 Rust stream ownership

**Files:**
- Create: `tinfer_rust/src/models/styletts2/stream.rs`
- Modify: `tinfer_rust/src/models/styletts2/mod.rs`
- Modify: `tinfer_rust/src/models/styletts2/preprocessing.rs`
- Modify: `tinfer_rust/src/models/styletts2/postprocessing.rs`
- Test: `tinfer_rust/src/models/styletts2/tests.rs`

**Interfaces:**
- `StreamState` stores prepared text, token durations, alignment, previous style, and whether native generation is active.
- Initial requests prepare A inputs and include `stream_ids` and `operation = start` tensors. Start never loops through B+C internally.
- Continuations include only `stream_ids` and `operation = continue`.

- [ ] Group initial calls by diffusion parameters as today; group continuation calls separately.
- [ ] Do not place starts and continuations in the same native batch.
- [ ] Store alignment metadata by `stream_id`, not in the generic engine.
- [ ] Decode native flattened audio using `audio_offsets`.
- [ ] Read `frame_starts`, `frame_counts`, and `complete` from native output.
- [ ] Return only alignments intersecting the emitted frame interval, rebased to the audio chunk.
- [ ] Preserve `previous_style_vector` and the last text/token context after a chunk completes.
- [ ] Reject starting a second chunk for a stream while its native session is active.
- [ ] Remove all Rust and native state in `close_stream`.
- [ ] Test output decoding and alignment slicing with constructed native tensors.

Native control tensors:

```text
operation:  I32 [1]       0=start, 1=continue, 2=close
stream_ids: I64 [batch]
```

Native result tensors:

```text
audio:         F32 [sum(samples)]
audio_offsets: I32 [batch + 1]
frame_starts:  I32 [batch]
frame_counts:  I32 [batch]
complete:      Bool [batch]
durations:     I32 [batch, token_width]   # start only
style:         F32 [batch, 256]           # start only
```

### Task 3: Native acoustic sessions

**Files:**
- Create: `tinfer_rust/src/models/styletts2/cpp/session.hpp`
- Create: `tinfer_rust/src/models/styletts2/cpp/window.hpp`
- Modify: `tinfer_rust/src/models/styletts2/cpp/model.hpp`
- Modify: `tinfer_rust/src/models/styletts2/cpp/model.cpp`
- Modify: `tinfer_rust/src/models/styletts2/cpp/cuda/pipeline.cpp`
- Modify: `tinfer_rust/build.rs`

**Interfaces:**
- `Session` owns token-level A outputs, predicted durations, total frame count, emitted cursor, seed, and retained frame context.
- `Window` always describes 64 pre, 48 core, and 16 post frames.
- Sessions are keyed by Rust engine stream ID inside `StyleTts2Model`.

- [ ] Split batched A outputs into session-owned token tensors without expanding all acoustic frames.
- [ ] Return from start immediately after creating sessions, leaving the first B+C window for a separately scheduled continuation.
- [ ] Store duration prefix sums and locate the token for any requested acoustic frame.
- [ ] Build each 128-frame B input lazily from token representations.
- [ ] Use retained context for negative positions at the start of a later text chunk; use zeros only at the real stream beginning.
- [ ] Execute B once and C once for the complete continuation batch.
- [ ] Crop samples `[64 * 600, (64 + valid_core) * 600)` for each item.
- [ ] Flatten differently sized final outputs and return `audio_offsets`.
- [ ] Advance each cursor by its valid core count and mark the item complete at its final frame.
- [ ] Keep only the last 64 expanded acoustic frames after a text chunk finishes.
- [ ] Erase full token tensors immediately when no longer needed.
- [ ] Fail clearly if a stream ID is missing, duplicated, or started twice.

The iteration is constant-memory with respect to generated duration:

```cpp
struct Window {
  static constexpr int pre = 64;
  static constexpr int core = 48;
  static constexpr int post = 16;
  static constexpr int frames = 128;
  int start;
  int valid_core;
};
```

### Task 4: CPU fixed-window path

**Files:**
- Create: `tinfer_rust/src/models/styletts2/cpp/cpu/pipeline.cpp`
- Modify: `tinfer_rust/src/models/styletts2/cpp/cpu/glue.hpp`
- Modify: `tinfer_rust/src/models/styletts2/cpp/cpu/glue.cpp`
- Modify: `tinfer_rust/build.rs`

**Interfaces:**
- Consume batches of session windows assembled as contiguous `[B, channels, 128]` arrays.
- Produce cropped core audio and update retained context.

- [ ] Move the current full-length CPU pipeline out of `model.cpp` into the CPU file.
- [ ] Implement lazy token-to-frame expansion for a requested absolute frame interval.
- [ ] Pad unavailable pre/post frames with zero only at actual stream boundaries.
- [ ] Run B and C at exactly 128 frames for every non-empty continuation.
- [ ] Use absolute sample offsets for deterministic noise across windows.
- [ ] Preserve harmonic phase at emitted-core boundaries.
- [ ] Run native glue tests and the StyleTTS2 Rust unit tests.

### Task 5: CUDA fixed-window path

**Files:**
- Modify: `tinfer_rust/src/models/styletts2/cpp/cuda/glue.hpp`
- Modify: `tinfer_rust/src/models/styletts2/cpp/cuda/kernels.cu`
- Modify: `tinfer_rust/src/models/styletts2/cpp/cuda/pipeline.cpp`

**Interfaces:**
- Session token tensors remain device-resident.
- Window assembly and harmonic generation execute on the model CUDA stream.

- [ ] Add a window expansion kernel accepting per-item token tensors, duration prefixes, and absolute frame starts.
- [ ] Assemble fixed `[B, *, 128]` B/C inputs without copying full A outputs to the host.
- [ ] Copy only duration totals and flattened cropped audio to the host.
- [ ] Seed noise from the absolute stream sample offset.
- [ ] Carry harmonic phase across emitted cores while recomputing overlap context.
- [ ] Keep TensorRT execution contexts and all session buffers on the existing CUDA context and stream.
- [ ] Validate `actual_frames` before converting it to 32-bit indexes.

### Task 6: Export fixed acoustic contracts

**Files:**
- Modify: `tools/styletts2_model_scripts/onnx_export.py`
- Modify: `tools/styletts2_model_scripts/tensorrt_export.py`
- Modify: `tools/styletts2_model_scripts/artifacts.py`
- Modify: `tools/styletts2_model_scripts/convert_model.py`
- Test: `tools/styletts2_model_scripts/tests/test_tensorrt_export.py`
- Test: `tools/styletts2_model_scripts/tests/test_artifacts.py`

**Interfaces:**
- A remains dynamic for batch and token count, with maximum token count 512.
- B and C compile for acoustic length 128; batch remains dynamic.
- Artifact metadata records token maximum and acoustic window shape, not a generated-length ceiling.

- [ ] Remove `max_frames` as a runtime generation ceiling.
- [ ] Export B `en[..., 128]`, C `asr[..., 128]`, `f0/noise[..., 256]`, and `har[..., 15361]`.
- [ ] Keep only batch dimensions dynamic for B and C.
- [ ] Update architecture identity and metadata with `{pre: 64, core: 48, post: 16}`.
- [ ] Keep ONNX and TensorRT compilation code separate.
- [ ] Run the exporter unit tests once.

### Task 7: Streaming integration verification

**Files:**
- Modify: `tinfer_rust/tests/grpc_integration.rs`
- Modify: `tinfer_rust/tests/web_integration.rs`
- Modify: `tinfer_rust/tests/engine.rs`

**Interfaces:**
- No server-specific scheduling logic; servers consume repeated `AudioChunk` values from `Stream`.

- [ ] Test two simultaneous streams with unequal text lengths and confirm the short stream ends first.
- [ ] Test append, receive audio, append again, finish, and confirm monotonic chunk indexes and text spans.
- [ ] Test empty incremental messages remain no-ops.
- [ ] Test cancellation during a continuation releases the pinned model entry.
- [ ] Test shutdown while continuations exist completes without waiting for further model calls.
- [ ] Run `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`.
- [ ] Run focused engine, gRPC, web, and StyleTTS2 tests.
- [ ] Inspect `git diff --check` and `git status --short`; leave all changes uncommitted.

## Self-review

- The engine knows only `continuation` and `complete`; token/frame policy remains model-specific.
- Fixed acoustic shapes permit batching streams at unrelated absolute positions.
- Sessions stay pinned to one loaded entry, preventing state migration and duplicate model/context loading.
- Token outputs are stored, while acoustic frames are expanded lazily, so generated duration does not determine memory use.
- Current-model approximation uses context windows; the same contract can later be used for streaming fine-tuning.
- Close, cancel, unload, shutdown, and inference failure all have explicit state-release paths.
