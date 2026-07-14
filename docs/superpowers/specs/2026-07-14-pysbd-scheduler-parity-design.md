# pySBD, Chunking, and Scheduling Parity Design

## Scope

Port pySBD commit `5905f13be4fc95f407b98392e0ec303617a33d86` to a standalone Rust crate and make
tinfer_rust match the current Python tinfer chunking and scheduling behavior. The port includes all 23 registered
languages, non-destructive segmentation, character spans, cleaning, PDF cleaning, errors, and regression cases.

The Python sources under `tinfer/tinfer/utils/text_chunker.py`, `tinfer/tinfer/core/request.py`,
`tinfer/tinfer/core/scheduler.py`, and `tinfer/tinfer/scheduler/worker_scheduler.py` are authoritative for engine
behavior. Compatibility does not preserve Python multiprocessing; it preserves observable stream state, chunks,
spans, triggers, batching, and priority while retaining Rust threads and `tx.send` communication.

## Standalone crate

`tinfer_rust/pysbd` is an independent Cargo package named `pysbd`. It has no Python runtime dependency. It uses
`fancy-regex` for Python-compatible lookaround and backreferences and `regex` for patterns supported by Rust's
linear engine. Regex compilation failures are explicit `Error::Regex` failures.

The public surface is small:

```rust
pub struct Options { pub clean: bool, pub doc_type: Option<DocType> }
pub enum DocType { Pdf }
pub struct TextSpan { pub text: String, pub start: usize, pub end: usize }
pub struct Segmenter { /* language and options */ }

impl Segmenter {
    pub fn new(language: &str, options: Options) -> Result<Self>;
    pub fn segment(&self, text: &str) -> Result<Vec<String>>;
    pub fn segment_spans(&self, text: &str) -> Result<Vec<TextSpan>>;
}
```

`segment` returns cleaned/destructive sentences when `clean` is true and whitespace-preserving sentences when it
is false. `segment_spans` is rejected with cleaning, exactly matching pySBD's invalid `clean + char_span`
combination. Empty input produces an empty vector. Offsets are UTF-8 byte offsets so spans slice Rust strings and
concatenate exactly to the input; copied tests also assert equivalent Unicode content and Python character offsets
where they differ.

## Internal layout

Shared structs live in `rules.rs`. Shared processing lives in focused files: `processor.rs`, `cleaner.rs`,
`abbreviation.rs`, `punctuation.rs`, `lists.rs`, and `text.rs`. There is no trait hierarchy. A `Language` enum and
central `Rules` struct select constants and the few explicit language hooks.

Each language remains a separate source file. To honor the repository's folder limit, files are grouped under
`lang/a_f`, `lang/g_l`, and `lang/m_z`; this is physical grouping only. `lang/common.rs` and `lang/standard.rs` contain upstream
shared rules. Language tests use the same grouping and remain one test module per language.

The upstream MIT license is copied as `tinfer_rust/pysbd/LICENSE`, and `NOTICE` identifies the pinned commit and
which sources/tests were translated.

## Parity tests

Every upstream assertion in `tests/lang`, `tests/regression`, `test_segmenter.py`, `test_cleaner.py`, and
`test_languages.py` is represented in Rust. Parameterized Python rows become table-driven Rust cases. Tests retain
the upstream input and expected output verbatim unless Rust escaping requires a syntax-only change.

The acceptance gate is:

```text
cargo test --manifest-path tinfer_rust/pysbd/Cargo.toml
```

It must report zero failures for shared, cleaner, segmenter, regression, and all 23 language suites.

## Tinfer chunking

`tinfer_rust/src/engine/chunker.rs` owns chunk construction and uses the local `pysbd` crate. It ports these Python
operations directly:

- schedule-derived `trigger` and `no_split_limit`, including last-entry extrapolation;
- sentence segmentation without losing whitespace;
- oversized splitting in order: blank lines, lines, sentence punctuation, comma/semicolon, spaces, hard limit;
- schedule-aware packing and final two-chunk merge;
- prepared-chunk buffering for single-in-flight streams;
- exact source spans across gaps and trailing whitespace.

All sizes use Unicode scalar counts, matching Python `len` for the supported corpora. Source positions stored in
`AudioChunk` remain byte ranges so they safely slice Rust input. Tests cover the conversion explicitly.

## Tinfer scheduling

The coordinator remains multithreaded and synchronous. `AsyncEngine` remains only a wrapper. All state stays in
the engine coordinator and all communication continues through channels.

Each request tracks creation, first text, generation-window deadline, first dispatch start, collected audio
seconds, prepared chunks, pending count, and force state. A request is dispatchable only when it has no in-flight
chunk and force, timeout, or `pending_length > trigger` fires. Force is consumed by the trigger decision.

Ready requests are grouped by model. Request score is the Python formula:

```text
(1_000_000_000 when now - start_time > collected_time else 0) + (now - created_at)
```

Requests sort descending by score. Each model contributes at most `max_batch` requests to a batch, and model
batches sort descending by their maximum request score. Stable insertion order breaks equal scores. Available
configured model entries execute eligible batches concurrently; each entry still has exactly one loaded model.

Successful audio adds `sample_count / sample_rate` to collected time. Inference error resets collected time and
start time. Cancellation invalidates queued/prepared work and wakes the stream. A completion with a stale request
or cancelled nonce is ignored rather than delivered.

## Integration and proof

Remove `sentencex` after the local crate is wired. Integration tests compare Rust chunk text and spans against
fixtures produced by current Python tinfer, and deterministic scheduler tests cover timeout, force, buffering,
priority, batch order, cancellation, errors, and multiple configured entries. Existing HTTP, WebSocket, gRPC,
normal-engine, and async-engine tests must remain green.

Implementation happens on `codex/pysbd-scheduler-parity`. Language subagents use isolated worktrees and a shared
Cargo target directory. Each subagent commits only its language modules and tests. Accepted commits are cherry-
picked into the integration branch; the main worktree receives the final accepted diff without a commit.
