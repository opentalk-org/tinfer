# Text-to-Speech API Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the six reference text-to-speech endpoints, expose compatible model metadata, and wire every endpoint into the Electron example.

**Architecture:** Keep `WebSocketServer` as the aiohttp host, but move request parsing, parameter mapping, response formatting, HTTP synthesis, and multi-context state into focused modules. Each WebSocket context owns an independent `TTSStream`; HTTP handlers share the same validated request and audio encoder. Electron normalizes gRPC and HTTP discovery into one catalog and converts every transport response into its existing playback event.

**Tech Stack:** Python 3.11, aiohttp, dataclasses, NumPy, Tinfer streams, Node.js, Electron, `@grpc/grpc-js`, native `fetch`, WebSocket, Node test runner, unittest.

## Global Constraints

- Work directly in `/workspace/tinfer`; do not create a worktree or commit.
- Run Python commands through `nix develop`.
- Preserve all unrelated uncommitted changes.
- Keep every changed file below 300 lines and every folder below 16 files.
- Implement only the contracts documented by the official [TTS endpoint index](https://elevenlabs.io/docs/api-reference/text-to-speech) and [model-list API](https://elevenlabs.io/docs/api-reference/models/list).
- The provider name may appear in compatibility documentation and source links only. It must not appear in Python, JavaScript, HTML, CSS, protobuf, test paths, identifiers, strings, or code comments.
- Do not add compatibility fallbacks. Invalid formats, languages, messages, and state transitions must fail explicitly.
- Use the official default model ID `eleven_multilingual_v2` when `model_id` is omitted; return not-found if it is not loaded.
- Validate output formats before calling the existing permissive audio-format parser.
- Encode raw float audio at the transport boundary. Do not ask the engine to concatenate or measure encoded audio.

---

### Task 1: Shared protocol schemas, mapping, and formatting

**Files:**
- Modify: `tinfer/tinfer/server/websocket/config.py`
- Create: `tinfer/tinfer/server/websocket/schemas.py`
- Create: `tinfer/tinfer/server/websocket/request_mapper.py`
- Create: `tinfer/tinfer/server/websocket/response_formatter.py`
- Test: `tmp_tests/tts_api/test_contract_types.py`

**Interfaces:**
- Produce `SpeechQuery`, `VoiceSettings`, `GenerationConfig`, and `SpeechRequest` frozen dataclasses.
- Produce `parse_query(request)`, `parse_speech_request(payload)`, and `map_stream_params(query, speech, alignment_type)`.
- Produce `format_model(info)`, `encode_chunk(chunk, output_format)`, `format_ws_audio(...)`, and `format_http_timing(...)`.

- [ ] Write failing tests proving required `text`, official model default, strict output-format validation, baked-language validation, `speed/alpha/beta` mapping, and all three alignment casings: WebSocket camelCase milliseconds, HTTP snake_case seconds, and multi-context `contextId`.
- [ ] Run `nix develop -c python -m unittest tmp_tests.tts_api.test_contract_types -v`; expect failures because the modules do not exist.
- [ ] Replace the unused class-shaped config with frozen dataclasses. Parse the reference request keys without raw untyped dictionaries at module boundaries. Accept documented voice fields; map `speed`, while explicitly recording `stability`, `similarity_boost`, `style`, and `use_speaker_boost` as accepted/no-effect capabilities. Preserve Tinfer `alpha` and `beta` as extensions.
- [ ] Define exact supported HTTP and WebSocket format enums. Convert `language_code` to `tts_params.language`; reject a language not listed in the selected model metadata.
- [ ] Format `/v1/models` entries as `model_id`, `name`, `can_do_text_to_speech`, `languages: [{language_id, name}]`, and the Tinfer extension `default_language`. Put the default first and do not emit the old `supported_languages` field or `{models: ...}` wrapper.
- [ ] Run the focused test again; expect PASS.

### Task 2: Four HTTP synthesis endpoints

**Files:**
- Create: `tinfer/tinfer/server/websocket/http_handler.py`
- Modify: `tinfer/tinfer/server/websocket/server.py`
- Test: `tmp_tests/tts_api/fakes.py`
- Test: `tmp_tests/tts_api/test_http_speech.py`
- Test: `tmp_tests/tts_api/test_http_streaming.py`

**Interfaces:**
- Register `POST /v1/text-to-speech/{voice_id}`.
- Register `POST /v1/text-to-speech/{voice_id}/with-timestamps`.
- Register `POST /v1/text-to-speech/{voice_id}/stream`.
- Register `POST /v1/text-to-speech/{voice_id}/stream/with-timestamps`.

- [ ] Add fakes for `AsyncStreamingTTS`, `TTSStream`, model metadata, raw `AudioChunk` objects, delayed chunks, and cleanup counters.
- [ ] Write failing aiohttp tests for binary unary audio, unary JSON timing, first streaming bytes before inference completion, ordered newline-delimited timing objects, disconnect cleanup, 404 unknown model/voice, 422 malformed request, and 503 while draining.
- [ ] Run `nix develop -c python -m unittest tmp_tests.tts_api.test_http_speech tmp_tests.tts_api.test_http_streaming -v`; expect missing-route failures.
- [ ] Implement shared parsing and generation in `SpeechHttpHandler`. Unary generation must merge raw samples and encode once. Plain streaming must write encoded bytes in chunk order. Timing streaming must write one compact JSON object plus `\n` per chunk with `audio_base64`, `alignment`, and `normalized_alignment`.
- [ ] Use `application/octet-stream` for unary binary audio, the selected audio MIME for raw streaming, `application/json` for unary timing, and `text/event-stream` for timing streaming. End-of-body is the HTTP streaming completion signal.
- [ ] Hold `HealthState` admission for the complete response lifetime. Close streams in `finally`; before response preparation return structured errors, and after preparation terminate the response and log the inference failure.
- [ ] Run the focused HTTP tests; expect PASS. Add one MP3/Opus decode check proving concatenated encoded stream chunks remain decodable.

### Task 3: Single-context WebSocket

**Files:**
- Create: `tinfer/tinfer/server/websocket/stream_context.py`
- Modify: `tinfer/tinfer/server/websocket/handler.py`
- Test: `tmp_tests/tts_api/test_websocket_single.py`

**Interfaces:**
- Produce `StreamContext`, which owns one `TTSStream`, audio-pump task, inactivity task, close-after-drain state, and serialized response callback.
- Keep `WebSocketHandler` responsible only for validating and dispatching the single-context message union.

- [ ] Write failing tests requiring first message `{"text":" "}`, later text ending in a space, `flush:true` generation without closure, `{text:""}` force/drain/final behavior, `{isFinal:true}`, `isFinal:false` audio messages, inactivity cleanup, and exactly-once stream closure.
- [ ] Run `nix develop -c python -m unittest tmp_tests.tts_api.test_websocket_single -v`; expect protocol failures against the current handler.
- [ ] Move audio pumping, encoding, alignment response construction, inactivity, and cleanup into `StreamContext`, shrinking `handler.py` below 300 lines.
- [ ] Make empty text a graceful EOS: force pending text, drain every audio chunk, send `{"isFinal":true}`, close the socket, cancel tasks, and close the Tinfer stream. Do not retain the current silent empty-message behavior.
- [ ] Serialize all socket writes, reject changed per-stream settings after initialization, and map malformed JSON/state errors to one error frame followed by closure.
- [ ] Run the focused single-WebSocket tests; expect PASS.

### Task 4: Multi-context WebSocket

**Files:**
- Create: `tinfer/tinfer/server/websocket/multi_context_handler.py`
- Modify: `tinfer/tinfer/server/websocket/server.py`
- Test: `tmp_tests/tts_api/test_websocket_multi.py`

**Interfaces:**
- Register `GET /v1/text-to-speech/{voice_id}/multi-stream-input`.
- Route client `context_id` to server `contextId` through `dict[str, StreamContext]`.

- [ ] Write failing tests with two interleaved contexts proving isolated text/settings/audio, ordered output through one writer, `flush:true`, keepalive `{text:"", context_id}`, `close_context:true`, ID reuse only after finalization, context inactivity, and `close_socket:true` waiting for every draining context.
- [ ] Run `nix develop -c python -m unittest tmp_tests.tts_api.test_websocket_multi -v`; expect the route/handler to be absent.
- [ ] Implement one outbound queue and writer task so context pumps never call `send_str` concurrently. Context creation allocates one independent Tinfer stream; live IDs cannot be reinitialized.
- [ ] On `close_context`, force and drain that context, emit `{"contextId": id, "isFinal": true}`, close it once, and remove it. On `close_socket`, gracefully finalize all contexts, await their pumps, then close the socket. On disconnect/error, cancel every task and close every stream.
- [ ] Run the focused multi-context tests; expect PASS.

### Task 5: Model discovery, routing, and health coverage

**Files:**
- Modify: `tinfer/tinfer/server/websocket/server.py`
- Modify: `tinfer/tinfer/server/websocket/__init__.py`
- Test: `tmp_tests/tts_api/test_models_api.py`
- Test: `tmp_tests/tts_api/test_health_admission.py`

**Interfaces:**
- `GET /v1/models` returns a top-level compatible model array with Tinfer's `default_language` extension.
- Existing `/v1/voices` remains available for the Electron catalog.

- [ ] Write failing tests for the top-level model shape, default-first language ordering, capability values, route uniqueness, and active-request accounting across unary, streamed HTTP, and both WebSocket lifetimes.
- [ ] Run the two focused test modules; expect the model-shape assertion to fail.
- [ ] Reduce `server.py` to lifecycle, health, route registration, model/voice discovery, and delegation. Register static suffix routes before the base `/{voice_id}` POST route. Return explicit 404/422/503 responses.
- [ ] Run the focused tests, then `nix develop -c python -m unittest discover -s tmp_tests/tts_api -v`; expect PASS.

### Task 6: Electron transport and catalog layer

**Files:**
- Modify: `examples/realtime_highlight_electron/main.js`
- Create: `examples/realtime_highlight_electron/catalog.js`
- Create: `examples/realtime_highlight_electron/grpc-client.js`
- Create: `examples/realtime_highlight_electron/api-client.js`
- Create: `examples/realtime_highlight_electron/synthesis-run.js`
- Modify: `examples/realtime_highlight_electron/styletts.proto`
- Test: `examples/realtime_highlight_electron/test/catalog.test.js`
- Test: `examples/realtime_highlight_electron/test/api-client.test.js`

**Interfaces:**
- Normalize both discovery APIs to `{id, voices, languages: [{id, name}], defaultLanguage}`.
- Expose modes `ws_single`, `ws_multi`, `post_audio`, `post_timing`, `stream_audio`, and `stream_timing` alongside the three gRPC modes.
- Convert every response into `{audioBase64, sampleRate, durationMs, alignments, firstByteMs}`.

- [ ] Write Node tests for gRPC `models`, the top-level HTTP model array, voice association, default-language preservation, exact six endpoint paths, request/query bodies, single/multi WebSocket messages, split-boundary raw streaming, and split-boundary newline JSON parsing.
- [ ] Run `cd examples/realtime_highlight_electron && node --test test/*.test.js`; expect module-not-found failures.
- [ ] Copy the server's `ModelInfo` protobuf shape into the Electron proto. Move gRPC operations out of `main.js`; synthesis requests must continue sending the selected model ID string, not a serialized `ModelInfo`.
- [ ] Implement the six API transports with `output_format=pcm_24000`. Single WebSocket sends init `{text:" "}` then text; Force sends `{text:" ", flush:true}`; End sends `{text:""}`. Multi mode uses one generated context per UI run and sends context-aware flush, close-context, then close-socket messages.
- [ ] Parse plain POST/stream bytes without alignment. Parse timestamp seconds into millisecond alignment events. Preserve partial fetch and NDJSON boundaries across network chunks.
- [ ] Keep `main.js` as Electron lifecycle and IPC orchestration below 300 lines. Run the Node tests; expect PASS.

### Task 7: Electron model/language UI and file split

**Files:**
- Modify: `examples/realtime_highlight_electron/index.html`
- Modify: `examples/realtime_highlight_electron/renderer.js`
- Create: `examples/realtime_highlight_electron/catalog-ui.js`
- Create: `examples/realtime_highlight_electron/playback.js`
- Modify: `examples/realtime_highlight_electron/package.json`
- Modify: `examples/realtime_highlight_electron/README.md`

**Interfaces:**
- Model changes rebuild Language from `model.languages` and select `defaultLanguage`.
- Timing modes enable highlighting; plain audio modes disable it while retaining playback/save.

- [ ] Remove hardcoded language options and add all six API mode labels. Write a DOM-level catalog test proving model changes replace the language options and select the baked default.
- [ ] Split selector/control logic into `catalog-ui.js` and audio/timeline/highlighting into `playback.js`; keep every changed file below 300 lines and the example root at no more than 15 files.
- [ ] Wire Send/Send chunk/Force/End controls only where meaningful: POST modes send complete text once; both WebSockets expose live controls; streaming POST means streamed output, not incremental text input.
- [ ] Add `test` and expanded `check` scripts, update run/API instructions, and run `npm test` plus `npm run check`; expect PASS.

### Task 8: Compatibility documentation and final verification

**Files:**
- Create: `docs/astro/src/content/docs/server/http.mdx`
- Modify: `docs/astro/src/content/docs/server/websocket.mdx`
- Create: `docs/astro/src/content/docs/server/api-compatibility.mdx`
- Modify: `docs/astro/astro.config.mjs`
- Modify: `docs/astro/src/content/docs/server/overview.mdx`
- Modify: `README.md`

- [ ] Document all six exact paths, request/response examples, single and multi WebSocket close semantics, the top-level model response, and the `default_language` extension.
- [ ] Add a six-row matrix with columns `Surface`, `Reference path`, `Transport`, `Audio response`, `Timing`, `Incremental input`, and `Tinfer status/limitations`. Separately list accepted/no-effect fields, Tinfer extensions, rejected language/format behavior, unsupported dictionaries/auth/history/stitching/SSML semantics, and deprecated reference fields.
- [ ] Run `nix develop -c python -m unittest discover -s tmp_tests/tts_api -v`, Electron `npm test && npm run check`, `git diff --check`, and scans for stale response shapes plus the forbidden provider name across `*.py`, `*.js`, `*.html`, `*.css`, and `*.proto`; expect tests/checks to pass and no forbidden code occurrences.
- [ ] Start the server with the established `nix develop` command. Exercise all six endpoints and `/v1/models` against loaded English and Polish models; verify audio is nonempty, timing tiles input, both WebSockets emit final events, multi-context IDs do not cross, Electron selects each model's default language, and health returns to zero active requests.
- [ ] Review `git status --short`; confirm all changes remain uncommitted and no unrelated files were altered.
