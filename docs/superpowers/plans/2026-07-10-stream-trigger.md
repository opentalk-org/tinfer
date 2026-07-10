# Stream Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `min_chars_trigger` and make streaming generation use a fixed timeout window, silent whitespace input drops, and immediate continuation of an already-split text snapshot.

**Architecture:** `TTSRequest` owns the current generation-window timestamp and prepared split chunks. `TTSStream` drops whitespace-only additions before state changes and asks the engine to schedule the first timeout. `TextChunker` consumes cached split continuations before evaluating normal triggers, while `StreamingTTS` resets and reschedules the window only when it queues inference.

**Tech Stack:** Python dataclasses, threads, `queue.Queue`, pysbd, gRPC, aiohttp WebSocket, Markdown/YAML configuration.

## Global Constraints

- Do not add tests.
- Whitespace-only `add_text()` calls are no-ops and never enter pending text.
- Additional text does not restart an active generation window.
- Queueing inference resets the timer.
- Chunks from one split snapshot continue immediately.
- Preserve unrelated user changes and staged state.

---

### Task 1: Request and stream trigger state

**Files:**
- Modify: `tinfer/tinfer/core/request.py`
- Modify: `tinfer/tinfer/core/stream.py`

**Interfaces:**
- Produces: `TTSRequest.generation_window_started_at: float | None`
- Produces: `PreparedTextChunk(text: str, text_span: tuple[int, int])`
- Produces: `TTSRequest.prepared_chunks: list[PreparedTextChunk]`
- Produces: `TTSRequest.append_text(text: str) -> bool`, returning whether a generation window started
- Consumes: `StreamingTTS.schedule_timeout(request: TTSRequest)` from Task 3

- [ ] **Step 1: Remove the minimum-character API and add explicit trigger state**

Remove `min_chars_trigger` from `StreamParams` and `TTSRequest`. Add the generation timestamp and prepared chunk list. Make `append_text()` return `False` without mutation for `not text.strip()`, start a window only when none exists, and return whether it started one.

- [ ] **Step 2: Replace the minimum gate with non-whitespace pending-text eligibility**

Make `should_trigger_now()` reject `not pending_text.strip()`, preserve force/schedule checks, and use `generation_window_started_at` for timeout elapsed time. Empty force requests must not arm a future request.

- [ ] **Step 3: Integrate stream additions with timeout scheduling**

In `TTSStream.add_text()`, return immediately for ignored input. When `append_text()` starts a window, call `engine.schedule_timeout(request)`, then signal input. Make `force_generate()` a no-op unless non-whitespace pending text exists.

### Task 2: Chunk snapshot continuation

**Files:**
- Modify: `tinfer/tinfer/utils/text_chunker.py`

**Interfaces:**
- Consumes: `TTSRequest.prepared_chunks`
- Produces: immediate chunks from the prepared snapshot before normal trigger evaluation

- [ ] **Step 1: Remove minimum-size merging**

Delete `_enforce_min_chars_trigger()` and its call. The chunk schedule remains the only splitting-size rule.

- [ ] **Step 2: Cache and drain split snapshots**

When `single_chunk=True` and splitting returns multiple chunks, retain every chunk after the first in `request.prepared_chunks`. Store trimmed synthesis text separately from its exact original source span so whitespace advances buffer/alignment positions without exceeding the synthesis schedule. On later calls, pop a prepared chunk before calling `should_trigger_now()` so it queues immediately after the preceding result.

- [ ] **Step 3: Fail clearly on impossible internal empty chunks**

After splitting, raise `ValueError("text chunker produced an empty chunk")` if any produced chunk lacks non-whitespace content. External whitespace input is already dropped by Task 1.

### Task 3: Engine timeout lifecycle

**Files:**
- Modify: `tinfer/tinfer/core/engine.py`

**Interfaces:**
- Produces: `StreamingTTS.schedule_timeout(request: TTSRequest) -> None`
- Consumes: `TTSRequest.generation_window_started_at`
- Consumes: `TTSRequest.prepared_chunks`

- [ ] **Step 1: Remove minimum-character propagation**

Delete `min_chars_trigger` from `_STREAM_PARAM_KEYS` and request construction.

- [ ] **Step 2: Schedule the initial fixed timeout window**

Add `schedule_timeout()` to enqueue the request's current generation-window timestamp. The timeout loop continues to call `signal_input()`; stale events remain harmless because eligibility uses the latest timestamp.

- [ ] **Step 3: Reset the timer only when inference is queued**

After committing a dispatched chunk, set `generation_window_started_at` to the queue timestamp when ordinary pending text remains, otherwise clear it. Schedule a timeout for remaining ordinary text. Do not delay `prepared_chunks`, because result processing calls `signal_input()` and the chunker drains them first.

- [ ] **Step 4: Reset cancellation state**

Clear prepared chunks and the generation-window timestamp during request cancellation.

### Task 4: Remove configuration and documentation surface

**Files:**
- Modify: `tinfer/tinfer/config/engine_config.py`
- Modify: `server/config.yml`
- Modify: `docs/astro/src/content/docs/api/streaming-tts-config.mdx`
- Modify: `docs/astro/src/content/docs/concepts/parameters.mdx`
- Modify: `docs/astro/src/content/docs/concepts/streaming.mdx`

**Interfaces:**
- Removes: all supported `min_chars_trigger` configuration and documentation

- [ ] **Step 1: Remove runtime configuration**

Delete the dataclass field and YAML key.

- [ ] **Step 2: Rewrite trigger documentation**

Document the non-whitespace gate, fixed non-restarting timeout window, queue-time reset, immediate split continuation, force behavior, and silent whitespace-only input drop. Remove stale minimum-schedule references in the affected trigger tables.

### Task 5: Verification without new tests

**Files:**
- Verify only; do not create or modify test files.

**Interfaces:**
- Consumes all preceding tasks.

- [ ] **Step 1: Run static compilation**

Run `python3 -m compileall -q tinfer/tinfer` and expect exit status 0.

- [ ] **Step 2: Scan the removed API**

Run `rg -n "min_chars_trigger" tinfer server docs/astro examples` and expect no results.

- [ ] **Step 3: Run direct scheduler probes**

Use a minimal import harness that stubs unavailable audio dependencies and exercise whitespace drops, a non-restarting timeout window, queue-time reset, force, schedule triggering, and prepared split continuation. Expect every assertion to pass.

- [ ] **Step 4: Inspect the scoped diff**

Run `git diff --check` and review only the files named in this plan, confirming unrelated worktree and staged changes remain intact.
