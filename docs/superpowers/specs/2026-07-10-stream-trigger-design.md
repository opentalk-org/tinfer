# Stream Trigger Design

## Goal

Remove the configurable minimum-character threshold. A stream may synthesize any
pending text containing at least one non-whitespace character while silently
ignoring whitespace-only input chunks.

## Input behavior

`TTSStream.add_text()` accepts whitespace-only input as a no-op. It does not add
the input to the request buffer, signal the engine, or change the generation
timer. This rule applies transitively to direct library, gRPC, and WebSocket
usage. Explicit force and flush transport controls remain independent of text
input.

## Trigger window

The first accepted text after an empty pending buffer starts the request's
generation window. Additional text joins the same window without restarting it.
The request becomes eligible when it exceeds the current chunk-length schedule,
is explicitly forced, or the configured timeout expires.

Queueing a synthesis chunk resets the generation-window timestamp. Empty
pending text is never eligible.

## Split continuations

When eligible pending text is split into multiple chunks, the chunker retains
the resulting snapshot. The first chunk is queued normally. Each remaining
chunk from that snapshot becomes eligible immediately after the preceding
inference completes. Text appended after the snapshot is not part of the
immediate continuation and waits for its own normal trigger window.

The chunker raises an explicit internal error if its splitting process produces
an empty chunk. Whitespace-only external input never reaches this point because
it is ignored by `add_text()`.

## API removal

Remove `min_chars_trigger` from stream parameter types, request state, engine
configuration, YAML configuration, runtime propagation, and public
documentation. Remove the chunk-level minimum-size merge pass so the
chunk-length schedule is the only size rule.

## Verification

Do not add tests. Run the repository's applicable existing checks and direct
scheduler probes covering whitespace no-ops, fixed timeout windows, immediate
length/force triggers, and immediate split continuations.
