# Derived Chunk Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive an immediate-generation trigger and a larger no-split limit from each existing chunk schedule entry so small overflow does not create tiny continuation chunks.

**Architecture:** `TTSRequest` exposes one typed `ChunkLimits` calculation from its schedule. Trigger evaluation consumes the current trigger, while `TextChunker` uses the no-split limit before applying sentence-aware splitting around the trigger target. No new configuration surface is introduced.

**Tech Stack:** Python dataclasses, existing streaming request/chunker state, Markdown documentation.

## Global Constraints

- Do not add tests.
- Do not add or remove public configuration fields.
- The current schedule entry remains the strict immediate-generation trigger and preferred split target.
- The next schedule entry is the inclusive no-split limit.
- Extrapolate final-entry headroom from the last increase; use one-third headroom for a single-entry schedule.
- Preserve unrelated working-tree and staged changes.

---

### Task 1: Centralize derived schedule limits

**Files:**
- Modify: `tinfer/tinfer/core/request.py:89-175`

**Interfaces:**
- Produces: `ChunkLimits(trigger: int, no_split_limit: int)`
- Produces: `TTSRequest.get_chunk_limits(chunk_index: int) -> ChunkLimits`
- Consumes: `TTSRequest.chunk_length_schedule`

- [ ] **Step 1: Add the typed limit result**

Define an immutable `ChunkLimits` dataclass beside `PreparedTextChunk` with integer `trigger` and `no_split_limit` fields.

- [ ] **Step 2: Derive limits from the existing schedule**

Implement `get_chunk_limits()` so an in-range next entry supplies headroom. At and beyond the final entry, use `trigger + max(1, trigger - previous)` for multi-entry schedules and `trigger + max(1, trigger // 3)` for a single-entry schedule.

- [ ] **Step 3: Use the derived trigger**

Replace the inline schedule-index calculation in `should_trigger_now()` with `get_chunk_limits(chunk_index).trigger`, retaining the strict `pending_length > trigger` boundary.

### Task 2: Apply the derived no-split limit

**Files:**
- Modify: `tinfer/tinfer/utils/text_chunker.py:49-190`

**Interfaces:**
- Consumes: `TTSRequest.get_chunk_limits()`
- Produces: unsplit text through the inclusive no-split limit

- [ ] **Step 1: Change the split decision**

In `split_text_if_needed()`, return one chunk when pending text length is less than or equal to `no_split_limit`; only enter sentence-aware splitting above it.

- [ ] **Step 2: Keep split packing anchored to the trigger target**

Rename `get_max_chunk_size()` to `get_target_chunk_size()` and make it return `get_chunk_limits(chunk_index).trigger`. Update oversized splitting and packing call sites to use that target.

### Task 3: Document schedule semantics

**Files:**
- Modify: `docs/astro/src/content/docs/api/streaming-tts-config.mdx:9-16`
- Modify: `docs/astro/src/content/docs/concepts/parameters.mdx:16-23`
- Modify: `docs/astro/src/content/docs/concepts/streaming.mdx:17-31`

**Interfaces:**
- Documents: trigger, preferred split target, derived no-split limit, final extrapolation, and immediate prepared continuations

- [ ] **Step 1: Update configuration and parameter tables**

Describe each chunk schedule entry as the immediate trigger/preferred split target and the next entry as its derived inclusive no-split limit.

- [ ] **Step 2: Add a concrete boundary example**

Document `[120, 160, 250, 290]` at index zero: 1–120 waits, 121–160 generates immediately without splitting, and 161+ generates immediately with splitting around 120.

### Task 4: Verify without adding tests

**Files:**
- Verify only; do not create or modify test files.

**Interfaces:**
- Consumes all preceding tasks.

- [ ] **Step 1: Compile and run the existing API test**

Run `python3 -m compileall -q tinfer/tinfer` and `python3 -m pytest -q tmp_tests/test_chunk_schedule_api.py`; both must exit zero.

- [ ] **Step 2: Probe multi-entry boundaries**

Directly verify schedule `[120, 160, 250, 290]` at index zero: 120 does not length-trigger, 121 does, 160 remains one chunk, and 161 splits with no five-character tail.

- [ ] **Step 3: Probe final and single-entry derivation**

Verify final-entry limits `(290, 330)` and single-entry limits `(120, 160)`, including their inclusive split boundaries.

- [ ] **Step 4: Review the scoped diff**

Run scoped `git diff --check`, scan for unintended configuration changes, and request an independent read-only review.
