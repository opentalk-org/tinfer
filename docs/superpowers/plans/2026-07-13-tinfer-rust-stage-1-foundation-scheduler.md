# Tinfer Rust Stage 1 Foundation and Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a CPU-only Rust engine with typed configuration, explicit/automatic placement, one scheduler actor, deterministic fake inference, and current chunk/timeout/cancellation behavior.

**Architecture:** Tokio tasks communicate with one `Scheduler` through bounded commands. Scheduler state is mutated only in its command loop; CPU jobs return completion commands. A fake model and fake device inventory make lifecycle, placement, batching, and streams testable before native dependencies exist.

**Tech Stack:** Rust 2024, Tokio, Serde, serde_yaml, thiserror, uuid, tracing, tokio-util, proptest.

## Global Constraints

- Apply every constraint in `2026-07-13-tinfer-rust-port.md`.
- Do not add ONNX Runtime, TensorRT, CUDA, Tonic, Axum, or PyO3 in this stage.
- The stage gate is a CPU-only `cargo test --workspace` plus a runnable fake-model example.
- Run Cargo commands from `/workspace/tinfer/tinfer_rust` and Git command blocks from `/workspace/tinfer`.

---

### Task 1: Crate and Typed Contracts

**Files:**
- Create: `tinfer_rust/Cargo.toml`
- Create: `tinfer_rust/src/lib.rs`
- Create: `tinfer_rust/src/main.rs`
- Create: `tinfer_rust/src/core/{mod.rs,ids.rs,request.rs,audio.rs,alignment.rs}`
- Create: `tinfer_rust/src/support/{mod.rs,error.rs}`
- Test: `tinfer_rust/tests/contracts.rs`

**Interfaces:**
- Produces: `ModelId`, `VoiceId`, `StreamId`, `RequestId`, `DeviceId`, `AlignmentKind`, `AudioChunk`, `EngineError`, and `type Result<T> = std::result::Result<T, EngineError>`.

- [ ] **Step 1: Write failing contract tests**

```rust
#[test]
fn alignment_values_are_wire_stable() {
    assert_eq!(AlignmentKind::Word.as_str(), "word");
    assert_eq!(AlignmentKind::Character.as_str(), "char");
    assert_eq!(AlignmentKind::Phoneme.as_str(), "phoneme");
    assert_eq!(AlignmentKind::None.as_str(), "none");
}

#[test]
fn identifiers_are_not_interchangeable() {
    let model = ModelId::parse("stub").unwrap();
    assert_eq!(model.as_str(), "stub");
    assert!(ModelId::parse("").is_err());
}
```

- [ ] **Step 2: Confirm the test fails**

Run: `cd tinfer_rust && cargo test --test contracts`

Expected: compile failure because `tinfer_rust` contracts do not exist.

- [ ] **Step 3: Implement the contracts and error enum**

```rust
pub enum EngineError {
    Configuration(String), Validation(String), Catalog(String),
    Overloaded, Cancelled, Inference(String), Shutdown,
}

pub struct AudioChunk {
    pub audio: Arc<[f32]>, pub sample_rate: u32, pub chunk_index: u64,
    pub text_span: Range<usize>, pub alignment: Option<Alignment>,
    pub request_id: RequestId,
}
```

Use private `Arc<str>` fields and validated `parse` constructors for string IDs. Implement explicit `Display`, Serde, and `as_str`; do not add stringly typed aliases.

- [ ] **Step 4: Verify and commit**

Run: `cargo fmt --all && cargo test --test contracts`

Expected: both tests pass.

```bash
git add tinfer_rust/Cargo.toml tinfer_rust/src tinfer_rust/tests/contracts.rs
git commit -m "feat(rust): establish typed runtime contracts"
```

### Task 2: Configuration and Placement

**Files:**
- Create: `tinfer_rust/src/config/{mod.rs,engine.rs,model.rs,server.rs}`
- Create: `tinfer_rust/src/engine/{mod.rs,placement.rs,inventory.rs}`
- Test: `tinfer_rust/tests/config_placement.rs`

**Interfaces:**
- Consumes: typed IDs and `EngineError` from Task 1.
- Produces: `Placement`, `ExecutionDevice`, `BackendKind`, `PlatformKind`, `DeviceInventory`, and `AutoPlacer::resolve(&ModelRequirements, &AutoPlacement) -> Result<Vec<Placement>>`.

- [ ] **Step 1: Write failing placement tests**

```rust
#[test]
fn auto_placement_is_stable_and_explicit() {
    let inventory = fake_inventory([(0, 24), (1, 48)]);
    let placements = AutoPlacer::new(inventory).resolve(&stub_requirements(), &auto(3)).unwrap();
    assert_eq!(device_ids(&placements), [1, 0, 1]);
    assert!(placements.iter().all(Placement::is_explicit));
}

#[test]
fn tensorrt_cpu_is_rejected() {
    assert!(Placement::validate(tensorrt_cpu()).is_err());
}
```

- [ ] **Step 2: Confirm the test fails**

Run: `cargo test --test config_placement`

Expected: compile failure for missing `AutoPlacer` and `Placement`.

- [ ] **Step 3: Implement validation and deterministic selection**

```rust
pub struct Placement {
    pub model: ModelId, pub artifact_dir: PathBuf, pub backend: BackendKind,
    pub platform: PlatformKind, pub device: ExecutionDevice,
    pub replicas: NonZeroUsize, pub slots_per_replica: NonZeroUsize,
    pub max_batch: NonZeroUsize,
}
```

Filter incompatible devices, sort GPUs by descending free bytes then ascending stable ID, distribute requested replicas round-robin, and return only fully explicit placements. Parse YAML with `deny_unknown_fields` and reject empty devices, zero capacities, TensorRT/CPU, and CUDA platform/CPU.

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test config_placement`

Expected: all validation and stable-order tests pass.

```bash
git add tinfer_rust/src/config tinfer_rust/src/engine tinfer_rust/tests/config_placement.rs
git commit -m "feat(rust): add explicit and automatic placement"
```

### Task 3: Chunker and Stream State

**Files:**
- Create: `tinfer_rust/src/core/{chunk.rs,stream.rs}`
- Create: `tinfer_rust/src/scheduling/{mod.rs,state.rs,timeout.rs}`
- Test: `tinfer_rust/tests/chunker_parity.rs`
- Test: `tinfer_rust/tests/fixtures/chunker_cases.json`

**Interfaces:**
- Produces: `ChunkSchedule::new(Vec<usize>)`, `TextChunker::next(&str, ChunkLimits, bool) -> Option<PreparedChunk>`, and private `StreamState` with UTF-8 byte spans.

- [ ] **Step 1: Extract and write parity fixtures**

Copy concrete inputs and outputs from `tmp_tests/test_chunk_schedule_api.py`, deleted test history if present in Git, and current `tinfer/tinfer/utils/text_chunker.py`. Store JSON cases for leading spaces, abbreviations, Unicode punctuation, repeated substrings, forced flush, and derived final limits.

```rust
#[test]
fn chunker_matches_reference_fixtures() {
    for case in fixtures() {
        assert_eq!(run_case(&case), case.expected);
    }
}
```

- [ ] **Step 2: Confirm fixture failure**

Run: `cargo test --test chunker_parity`

Expected: compile failure for missing `TextChunker`.

- [ ] **Step 3: Port chunking exactly**

Implement positive nondecreasing schedules, next-entry no-split limits, derived final limits, sentence then punctuation then word boundaries, hard-limit splitting, separator preservation, and source byte ranges. Keep text normalization out of this module.

```rust
pub struct PreparedChunk { pub text: String, pub source: Range<usize> }
pub struct ChunkLimits { pub trigger: usize, pub no_split: usize }
```

- [ ] **Step 4: Verify and commit**

Run: `cargo test --test chunker_parity`

Expected: every reference fixture passes.

```bash
git add tinfer_rust/src/core tinfer_rust/src/scheduling tinfer_rust/tests
git commit -m "feat(rust): port text chunking and stream state"
```

### Task 4: Unified Scheduler and Fake Executor

**Files:**
- Create: `tinfer_rust/src/scheduling/{scheduler.rs,queue.rs,priority.rs}`
- Create: `tinfer_rust/src/engine/{engine.rs,catalog.rs,lifecycle.rs}`
- Create: `tinfer_rust/src/models/{mod.rs,traits.rs,registry.rs,fake.rs}`
- Test: `tinfer_rust/tests/scheduler.rs`
- Test: `tinfer_rust/examples/fake_server.rs`

**Interfaces:**
- Produces: `Engine::start`, `Engine::load_model`, `Engine::create_stream`, `StreamHandle::{append,trigger,flush,cancel,recv}`, `ModelFactory`, `ModelReplica`, and scheduler `Command`.

- [ ] **Step 1: Write scheduler behavior tests**

```rust
#[tokio::test(start_paused = true)]
async fn timeout_keeps_first_window_deadline() { /* append twice, advance, assert one dispatch */ }

#[tokio::test]
async fn one_stream_is_sequential_but_streams_overlap() { /* fake slot barriers assert ordering */ }

#[tokio::test]
async fn cancelled_generation_drops_late_completion() { /* cancel after dispatch, release fake job */ }
```

Make the test bodies use fake-model barriers and assert concrete dispatch and chunk sequences; also cover batching keys, max batch, starvation boost, bounded overload, stable replica choice, and ordered delivery.

- [ ] **Step 2: Confirm failure**

Run: `cargo test --test scheduler`

Expected: compile failure for missing `Engine` and `Scheduler`.

- [ ] **Step 3: Implement the single actor**

```rust
enum Command {
    Create(CreateStream, oneshot::Sender<Result<StreamHandleParts>>),
    Append(StreamId, String), Trigger(StreamId), Flush(StreamId),
    Cancel(StreamId), Close(StreamId), Timeout(StreamId, u64),
    Completed(CompletedJob), Shutdown(oneshot::Sender<()>),
}
```

Use one bounded command receiver and one owned `SchedulerState`. Never await work from the command loop: lease a slot, move an immutable job to a bounded executor, and receive `Completed`. Use `DelayQueue`, monotonic cancellation generations, compatibility batch keys, age plus playback-starvation priority, and stable load/slot/ID selection.

- [ ] **Step 4: Verify the stage and commit**

Run: `cargo test --workspace && cargo run --example fake_server`

Expected: tests pass; example prints two ordered nonempty fake chunks then exits cleanly.

```bash
git add tinfer_rust/src tinfer_rust/tests/scheduler.rs tinfer_rust/examples/fake_server.rs
git commit -m "feat(rust): implement unified scheduler and engine"
```
