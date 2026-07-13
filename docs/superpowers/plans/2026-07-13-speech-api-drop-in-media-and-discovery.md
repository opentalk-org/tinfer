# Speech API Drop-in Compatibility Implementation Plan — Media and Discovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete drop-in compatibility for text processing, alignments, audio formats, discovery, Electron, documentation, and live verification after Tasks 1–5.

**Architecture:** Reuse the typed contracts, exact error formatters, generation state machine, and capability mapper from the contracts-and-generation plan. Add source-preserving preprocessing, response-scoped encoding, standard discovery formatters, and shared end-to-end fixtures.

**Tech Stack:** Python 3.11, aiohttp, dataclasses, NumPy, PyTorch, pydub/FFmpeg, Electron/Node test runner, unittest, Nix development shell.

## Global Constraints

- Complete Tasks 1–5 in `docs/superpowers/plans/2026-07-13-speech-api-drop-in-compatibility.md` first.
- Work directly in `/workspace/tinfer`; do not create a worktree or commits.
- Run Python commands through `nix develop` with the required espeak library path.
- Target source code, tests, and comments must not contain the provider name.
- Accept and silently drop every documented unsupported option; document it as accepted/no-effect.
- Expose only reference error/status/frame/close behavior.
- Keep every changed file under 300 lines and every changed folder under 16 files.
- Keep imports at module top and use typed structures at API boundaries.

---

### Task 6: Text normalization, SSML, continuity, and alignments

**Files:**
- Create: `tinfer/tinfer/server/websocket/text_options.py`
- Modify: `tinfer/tinfer/models/impl/styletts2/model/phonemizer.py`
- Modify: `tinfer/tinfer/models/impl/styletts2/alignment/converter.py`
- Modify: `tinfer/tinfer/core/request.py`
- Test: `tmp_tests/tts_api/test_text_options.py`
- Modify: `tmp_tests/test_alignment_spans.py`

**Interfaces:**
- Produces: `prepare_text(text: str, options: TextOptions) -> PreparedSpeechText`
- Produces: `PreparedSpeechText(source: str, normalized: str, spans: tuple[TextSpan, ...])`

- [ ] **Step 1: Add failing tests for normalization `auto/on/off`, supported SSML text extraction, ignored unsupported tags, previous/next text acceptance, and distinct original/normalized alignments.**

- [ ] **Step 2: Run text-option and alignment tests; expect normalization controls and normalized alignment to be missing or duplicated.**

- [ ] **Step 3: Implement one preprocessing result that preserves source-to-normalized spans. Use previous/next text only where the model exposes continuity context; otherwise accept and drop them.**

```python
@dataclass(frozen=True)
class TextSpan:
    source_start: int
    source_end: int
    normalized_start: int
    normalized_end: int

@dataclass(frozen=True)
class PreparedSpeechText:
    source: str
    normalized: str
    spans: tuple[TextSpan, ...]
```

- [ ] **Step 4: Resolve pronunciation locators only when a local resolver is configured; otherwise accept and drop them. Ensure unsupported SSML elements contribute their text content without executing directives.**

- [ ] **Step 5: Run alignment, phonemizer, text-option, and the original clipping regression tests; expect exact source reconstruction and separate normalized sequences.**

### Task 7: WAV and response-scoped compressed encoders

**Files:**
- Create: `tinfer/tinfer/server/websocket/audio_encoder.py`
- Modify: `tinfer/tinfer/server/websocket/schemas.py`
- Modify: `tinfer/tinfer/server/websocket/response_formatter.py`
- Modify: `tinfer/tinfer/server/websocket/http_handler.py`
- Modify: `tinfer/tinfer/server/websocket/stream_context.py`
- Test: `tmp_tests/tts_api/test_audio_formats.py`

**Interfaces:**
- Produces: `ResponseAudioEncoder(output_format: SpeechOutputFormat)` with `encode(chunk) -> bytes` and `finish() -> bytes`

- [ ] **Step 1: Add failing tests for seven unary WAV formats and complete concatenated decode of multi-chunk MP3 and Opus responses.**

```python
payload = b"".join(response_chunks)
decoded = AudioSegment.from_file(io.BytesIO(payload), format="mp3")
self.assertGreater(len(decoded), 0)
```

- [ ] **Step 2: Run the format tests; expect WAV rejection or independent compressed chunk encoding.**

- [ ] **Step 3: Implement WAV headers for unary output and one encoder lifecycle per HTTP stream or WebSocket context. Emit trailer bytes from `finish()` before the final frame or EOF.**

- [ ] **Step 4: Run all format, HTTP streaming, WSS, and live decode tests for PCM, MP3, Opus, WAV, μ-law, and A-law.**

### Task 8: Standard model and voice discovery

**Files:**
- Modify: `tinfer/tinfer/server/websocket/response_formatter.py`
- Modify: `tinfer/tinfer/server/websocket/server.py`
- Modify: `tinfer/tinfer/core/request.py`
- Modify: `tinfer/tinfer/core/engine.py`
- Create: `tinfer/tinfer/core/catalog.py`
- Modify: `tinfer/tinfer/server/grpc/service.py`
- Create: `tinfer/tinfer/server/grpc/catalog_service.py`
- Test: `tmp_tests/tts_api/test_models_api.py`
- Create: `tmp_tests/tts_api/test_voices_api.py`

**Interfaces:**
- Extends: `ModelInfo` with typed capability metadata and default language
- Produces: standard top-level model array and standard `{voices:[...]}` envelope
- Produces: `ModelCatalog.model_infos() -> tuple[ModelInfo, ...]`
- Produces: `ModelCatalog.voice_infos(model_id: str | None) -> tuple[VoiceInfo, ...]`

- [ ] **Step 1: Add failing golden tests for every model response field, voice envelope field, baked language ordering, and default model/voice association.**

- [ ] **Step 2: Run discovery tests; expect sparse model data and the custom voice objects.**

- [ ] **Step 3: Move catalog lookup and formatting into `core/catalog.py` and gRPC catalog response construction into `grpc/catalog_service.py`. Populate defensible local capabilities, use neutral empty/false values for unavailable optional metadata, preserve `default_language`, and leave the modified engine/service files below 300 lines.**

```python
@dataclass(frozen=True)
class VoiceInfo:
    voice_id: str
    name: str
    model_id: str
    languages: tuple[str, ...]
```

- [ ] **Step 4: Regenerate protobuf output only if the typed capability fields change the gRPC schema, then run discovery and gRPC tests.**

### Task 9: Electron contract and exhaustive documentation

**Files:**
- Modify: `examples/realtime_highlight_electron/api-client.js`
- Modify: `examples/realtime_highlight_electron/catalog.js`
- Modify: `examples/realtime_highlight_electron/test/api-client.test.js`
- Modify: `docs/astro/src/content/docs/server/api-compatibility.mdx`
- Modify: `docs/astro/src/content/docs/server/http.mdx`
- Modify: `docs/astro/src/content/docs/server/websocket.mdx`

**Interfaces:**
- Consumes: exact six-route contract, standard discovery responses, and documented Tinfer extensions

- [ ] **Step 1: Add failing Electron fixtures for null options, default model resolution, mapped controls, no-effect options, multi reinitialization, exact errors, WAV unary output, and complete compressed streams.**

- [ ] **Step 2: Run `. /opt/nvm/nvm.sh && npm test`; verify fixtures expose any stale request or response assumptions.**

- [ ] **Step 3: Update Electron request builders and decoders without adding provider names to source or comments. Update the matrix so every field says implemented, mapped, accepted/no-effect, or extension for each runtime.**

- [ ] **Step 4: Run Electron tests/checks and the Astro build; expect zero failures and every changed file below 300 lines.**

### Task 10: Full verification and live six-route acceptance

**Files:**
- Modify only files required by failures found in this task

- [ ] **Step 1: Run backend suites.**

Run: `LD_LIBRARY_PATH=/nix/store/cg2s5xhv346wyy9lfxk9m53g9rra9xhm-espeak-ng-1.52.0.1-unstable-2025-09-09/lib nix develop -c python -m unittest discover -s tmp_tests/tts_api -v`

Run: `LD_LIBRARY_PATH=/nix/store/cg2s5xhv346wyy9lfxk9m53g9rra9xhm-espeak-ng-1.52.0.1-unstable-2025-09-09/lib nix develop -c python -m unittest discover -s tmp_tests/styletts2 -v`

Run: `LD_LIBRARY_PATH=/nix/store/cg2s5xhv346wyy9lfxk9m53g9rra9xhm-espeak-ng-1.52.0.1-unstable-2025-09-09/lib nix develop -c python -m unittest tmp_tests.test_alignment_spans -v`

- [ ] **Step 2: Run Electron and documentation verification.**

Run in `examples/realtime_highlight_electron`: `. /opt/nvm/nvm.sh && npm test && npm run check`

Run in `docs/astro`: `. /opt/nvm/nvm.sh && npm run build`

- [ ] **Step 3: Restart the server from the final working tree and call all four HTTP and both WebSocket routes with defaulted, mapped, and no-effect fields. Decode every returned audio stream and validate timing lengths.**

- [ ] **Step 4: Verify repository constraints.**

Run: `git diff --check`

Run: `rg -i "elevenlabs" tinfer examples/realtime_highlight_electron tmp_tests/tts_api`

Expected: no matches in target code/tests, no whitespace errors, no changed file above 300 lines, and no changed folder above 16 files.

- [ ] **Step 5: Leave the verified server running on ports 8000 and 50051 and report all no-effect fields explicitly. Do not commit.**
