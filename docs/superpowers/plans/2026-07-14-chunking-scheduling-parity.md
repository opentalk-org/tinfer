# Tinfer Chunking and Scheduling Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace simplified Rust chunking/scheduling with observable parity to current Python tinfer.

**Architecture:** `engine/chunker.rs` owns pure chunk construction through the local `pysbd` crate. The existing
synchronous coordinator owns request timing/state and ranks ready batches before sending calls to caller threads.

**Tech Stack:** Rust 2024, crossbeam channels, local `pysbd`, deterministic integration tests.

## Global Constraints

- Python `text_chunker.py`, `request.py`, `core/scheduler.py`, and `worker_scheduler.py` are authoritative.
- Keep the engine multithreaded, not async; `AsyncEngine` only wraps it.
- Keep scheduling centralized and communication through `tx.send`.
- Preserve maximum concurrency and multiple config entries without loading duplicate models implicitly.
- Keep files under 300 lines and folders under 16 files; no compatibility fallback or speculative abstraction.

---

### Task 1: Exact chunker

**Files:**
- Create: `tinfer_rust/src/engine/chunker.rs`
- Create: `tinfer_rust/tests/chunker.rs`
- Modify: `tinfer_rust/Cargo.toml`, `tinfer_rust/src/engine/mod.rs`

**Interfaces:**
- Consumes: `pysbd::Segmenter`.
- Produces: `Chunker::new(language)`, `Chunker::next(text, offset, index, schedule) -> Result<Vec<PreparedChunk>>`.

- [ ] Convert current Python fixtures plus derived-limit, oversized separator, packing, repeated text, Unicode, and
trailing-whitespace cases into assertions over `PreparedChunk { text, range }`.
- [ ] Run `cargo test -p tinfer_rust --test chunker`; expect failures while the module is absent.
- [ ] Implement `limits`: next schedule entry is no-split; last entry extrapolates prior delta or one-third.
- [ ] Port `_sentence_chunks`, `_split_oversized`, `_split_keep_separator`, `_split_by_derived_limits`,
`_pack_to_schedule`, and `_prepare_source_spans` in Python call order.
- [ ] Run chunker and local pySBD tests; expect exact text/range parity.
- [ ] Commit `feat(engine): port tinfer text chunker`.

### Task 2: Request trigger and prepared-chunk state

**Files:**
- Modify: `tinfer_rust/src/engine/scheduler.rs`
- Test: `tinfer_rust/tests/engine.rs`, `tinfer_rust/tests/chunker.rs`

**Interfaces:**
- Consumes: Task 1 `PreparedChunk`.
- Produces request fields `prepared`, `created_at`, `generation_started_at`, `start_time`, `collected_time`.

- [ ] Add tests proving force is consumed once, timeout begins only on empty-to-nonempty transition, a pending model
call blocks another chunk, and prepared chunks dispatch sequentially without recomputation.
- [ ] Replace simplified `ready/next_chunk` with Python `should_trigger_now` and `single_chunk=true` flow.
- [ ] Commit ranges only when dispatching; restart timeout for remaining unprepared pending text.
- [ ] Run engine/chunker tests; expect zero failures.
- [ ] Commit `feat(engine): match tinfer stream trigger state`.

### Task 3: Worker priority and batch ordering

**Files:**
- Modify: `tinfer_rust/src/engine/scheduler.rs`, `tinfer_rust/src/engine/registry.rs`
- Create: `tinfer_rust/tests/scheduler_priority.rs`

**Interfaces:**
- Produces: pure `priority(now, created, start, collected) -> DurationScore` and descending stable batch selection.

- [ ] Test age ordering, `1e9` underrun boost, stable ties, per-model max batch, descending model-batch score, busy
entries, and two configured entries serving two eligible batches.
- [ ] Implement score as `underrun_boost + elapsed_since_created`, matching Python strict `>` underrun comparison.
- [ ] Build candidates per model, stable-sort requests and model batches descending, then acquire available registry
entries and send calls in that order.
- [ ] Run priority and engine tests; expect exact order and batch membership.
- [ ] Commit `feat(engine): match tinfer worker scheduling priority`.

### Task 4: Completion, error, and cancellation accounting

**Files:**
- Modify: `tinfer_rust/src/engine/scheduler.rs`, `tinfer_rust/src/engine/engine.rs`
- Test: `tinfer_rust/tests/engine.rs`, `tinfer_rust/tests/scheduler_priority.rs`

**Interfaces:**
- Consumes: model output sample count/rate and dispatch generation nonce.
- Produces: exact collected-time update/reset and stale completion suppression.

- [ ] Test successful audio accumulation, inference reset, cancel clearing prepared work, stale result suppression,
unload wakeup, and finish/end delivery with queued chunks.
- [ ] Add a generation nonce to dispatch metadata; invalidate it on cancel and ignore mismatches.
- [ ] Add audio duration after success; reset `collected_time` and `start_time` on error.
- [ ] Run engine, async engine, HTTP, WebSocket, and gRPC integration tests.
- [ ] Commit `fix(engine): match tinfer completion lifecycle`.

### Task 5: Dependency removal and final proof

**Files:**
- Modify: `tinfer_rust/Cargo.toml`, `tinfer_rust/Cargo.lock`
- Modify only integration defects found by tests.

**Interfaces:**
- Produces: final accepted parity implementation.

- [ ] Remove `sentencex` and confirm `rg 'sentencex' tinfer_rust` has no matches.
- [ ] Run pySBD full tests, tinfer_rust default tests with codec libraries, formatting, and Clippy `-D warnings`.
- [ ] Verify file/folder limits and inspect the complete branch diff for unrelated changes.
- [ ] Commit `test(engine): prove chunking and scheduling parity`.
