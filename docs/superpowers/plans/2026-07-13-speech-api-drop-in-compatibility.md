# Speech API Drop-in Compatibility Implementation Plan — Contracts and Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standard clients use Tinfer's six speech routes and discovery APIs without request changes, while mapping meaningful controls and silently dropping documented unsupported options.

**Architecture:** A provider-neutral contract boundary parses the reference wire schema into typed models, classifies fields as implemented, mapped, no-effect, or extension, and converts supported values into Tinfer stream parameters. Shared HTTP and WebSocket generation components own state transitions, response-scoped encoding, alignments, and exact transport errors.

**Tech Stack:** Python 3.11, aiohttp, dataclasses/enums, NumPy, PyTorch, pydub/FFmpeg, Electron/Node test runner, unittest, Nix development shell.

## Global Constraints

- Work directly in `/workspace/tinfer`; do not create a worktree.
- Do not create commits.
- Run Python commands through `nix develop` with the required espeak library path.
- Target source code, tests, and comments must not contain the provider name, including comments and identifiers.
- Every documented syntactically valid field is accepted; unsupported fields are silently ignored and documented as accepted/no-effect.
- Invalid input and runtime failures use only the reference API's status, body, frame, and close behavior.
- Keep every changed file under 300 lines and every changed folder under 16 files.
- Imports stay at module top; structured data uses dataclasses, enums, or typed models rather than raw dictionaries at boundaries.
- Package initializers export only public server entry points; do not add contract re-export modules.

---

### Task 1: Typed contract and exact request acceptance

**Files:**
- Modify: `tinfer/tinfer/server/websocket/schemas.py`
- Create: `tinfer/tinfer/server/websocket/query_parser.py`
- Create: `tinfer/tinfer/server/websocket/speech_parser.py`
- Delete: `tinfer/tinfer/server/websocket/request_mapper.py`
- Modify: `tinfer/tinfer/server/websocket/handler.py`
- Modify: `tinfer/tinfer/server/websocket/multi_context_handler.py`
- Modify: `tinfer/tinfer/server/websocket/http_handler.py`
- Test: `tmp_tests/tts_api/test_contract_types.py`

**Interfaces:**
- Produces: `parse_query(request: Any, transport: Transport) -> SpeechQuery`
- Produces: `parse_speech_request(payload: Mapping[str, Any]) -> SpeechRequest`
- Produces: `map_stream_params(query: SpeechQuery, speech: SpeechRequest, alignment_type: AlignmentType) -> StreamParams`

- [ ] **Step 1: Write failing table tests for every documented query/body field, explicit null, enum, bound, and no-effect field.**

```python
def test_documented_no_effect_fields_are_accepted():
    request = parse_speech_request({
        "text": "Hello",
        "voice_settings": None,
        "pronunciation_dictionary_locators": [{
            "pronunciation_dictionary_id": "dictionary",
            "version_id": "version",
        }],
        "previous_text": "Before",
        "next_text": "After",
        "previous_request_ids": ["previous"],
        "next_request_ids": ["next"],
    })
    assert request.text == "Hello"
    assert request.voice_settings.speed is None
```

- [ ] **Step 2: Run the contract test and verify current parsing rejects the documented fields.**

Run: `LD_LIBRARY_PATH=/nix/store/cg2s5xhv346wyy9lfxk9m53g9rra9xhm-espeak-ng-1.52.0.1-unstable-2025-09-09/lib nix develop -c python -m unittest tmp_tests.tts_api.test_contract_types -v`

Expected: FAIL on `voice_settings: null` or the first documented no-effect field.

- [ ] **Step 3: Split query and speech parsing and retain ignored values in typed contract fields only when later behavior needs them.**

```python
class Transport(Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"

@dataclass(frozen=True)
class ContinuitySettings:
    previous_text: str | None
    next_text: str | None
    previous_request_ids: tuple[str, ...]
    next_request_ids: tuple[str, ...]

voice_payload = {} if payload["voice_settings"] is None else payload["voice_settings"]
```

Unknown JSON fields remain validation errors because the reference schema does not accept them. Known unsupported fields parse successfully and do not enter `StreamParams`.

- [ ] **Step 4: Replace all `request_mapper` imports with the defining parser modules and delete the old file.**

- [ ] **Step 5: Run all contract tests and confirm every documented valid field passes while invalid types and enum members produce the expected validation failure.**

### Task 2: Default model resolution and discovery-compatible aliases

**Files:**
- Create: `tinfer/tinfer/server/websocket/model_resolver.py`
- Modify: `tinfer/tinfer/server/websocket/server.py`
- Modify: `tinfer/tinfer/server/websocket/http_handler.py`
- Modify: `tinfer/tinfer/server/websocket/handler.py`
- Modify: `tinfer/tinfer/server/websocket/multi_context_handler.py`
- Test: `tmp_tests/tts_api/test_models_api.py`

**Interfaces:**
- Produces: `ModelResolver.resolve(requested_id: str | None) -> ModelInfo`
- Produces: `ModelResolver.external_id(info: ModelInfo) -> str`

- [ ] **Step 1: Add failing tests proving omitted `model_id` resolves to the configured default loaded model and explicit IDs remain exact.**

```python
resolver = ModelResolver([ModelInfo("libri", ("en-us",), "en-us")], "libri")
assert resolver.resolve(None).model_id == "libri"
assert resolver.resolve("missing")  # raises the reference unknown-model error
```

- [ ] **Step 2: Run `tmp_tests.tts_api.test_models_api` and verify omission currently resolves to an unavailable fixed ID.**

- [ ] **Step 3: Implement one resolver instance in `WebSocketServer`, pass it to all handlers, and expose the configured default consistently through model discovery.**

```python
class ModelResolver:
    def __init__(self, infos: list[ModelInfo], default_model_id: str) -> None:
        self._by_id = {info.model_id: info for info in infos}
        self._default_model_id = default_model_id

    def resolve(self, requested_id: str | None) -> ModelInfo:
        model_id = requested_id if requested_id is not None else self._default_model_id
        return self._by_id[model_id]
```

- [ ] **Step 4: Run model, HTTP, and WebSocket tests; expect omission and discovery to select the same model.**

### Task 3: Exact error contract and failure propagation

**Files:**
- Create: `tinfer/tinfer/server/websocket/errors.py`
- Modify: `tinfer/tinfer/server/websocket/server.py`
- Modify: `tinfer/tinfer/server/websocket/http_handler.py`
- Modify: `tinfer/tinfer/server/websocket/stream_context.py`
- Test: `tmp_tests/tts_api/test_errors.py`
- Test: `tmp_tests/tts_api/fakes.py`

**Interfaces:**
- Produces: `validation_response(issues: tuple[ValidationIssue, ...]) -> web.Response`
- Produces: `resource_response(kind: ResourceKind, identifier: str) -> web.Response`
- Produces: `close_for_failure(ws: WebSocketResponse, failure: SpeechFailure) -> Awaitable[None]`

- [ ] **Step 1: Add golden failing tests for missing fields, invalid enums, unknown model/voice, admission rejection, inference failure, and encoder failure before and after stream preparation.**

```python
self.assertEqual(response.status, 422)
self.assertEqual(await response.json(), {
    "detail": [{"loc": ["body", "text"], "msg": "Field required", "type": "missing"}]
})
```

- [ ] **Step 2: Run `tmp_tests.tts_api.test_errors`; expect failures against the current custom `{detail:{status,message}}` and WSS `{error:...}` variants.**

- [ ] **Step 3: Implement typed failures and one formatter per transport. Preserve chunk errors during unary merge and never write a custom error payload after streaming headers.**

```python
@dataclass(frozen=True)
class ValidationIssue:
    loc: tuple[str | int, ...]
    msg: str
    type: str

failed = next((chunk.error for chunk in chunks if chunk.error is not None), None)
if failed is not None:
    raise InferenceError(failed)
```

- [ ] **Step 4: Run error, health-admission, HTTP, and WSS lifecycle tests; expect exact golden responses and zero leaked active connections.**

### Task 4: Generation state compatibility

**Files:**
- Modify: `tinfer/tinfer/core/stream.py`
- Modify: `tinfer/tinfer/server/websocket/stream_context.py`
- Modify: `tinfer/tinfer/server/websocket/handler.py`
- Modify: `tinfer/tinfer/server/websocket/multi_context_handler.py`
- Test: `tmp_tests/tts_api/test_generation_state.py`

**Interfaces:**
- Produces: `TTSStream.try_generate() -> None`
- Produces: `StreamContext.reinitialize(stream, output_format, inactivity_timeout) -> Awaitable[None]`

- [ ] **Step 1: Add failing state tests for schedule thresholds, `auto_mode`, conditional trigger, unconditional flush, live-context reinitialization, keepalive, close ordering, and context-isolated failure.**

```python
await ws.send_json({"text": "Below threshold ", "try_trigger_generation": True})
self.assertEqual(stream.force_count, 0)
await ws.send_json({"text": "", "flush": True})
self.assertEqual(stream.force_count, 1)
```

- [ ] **Step 2: Run the generation-state tests; verify trigger currently forces and live multi-context settings currently fail.**

- [ ] **Step 3: Add `try_generate()` as a scheduler signal without setting `force_next_generation`; map `auto_mode=true` to `timeout_trigger_ms=0`; atomically finalize and replace a reinitialized context.**

```python
def try_generate(self) -> None:
    self._engine.signal_input()

params["timeout_trigger_ms"] = 0.0 if query.auto_mode else DEFAULT_TIMEOUT_MS
```

- [ ] **Step 4: Run single, multi, generation-state, and text-chunker tests; expect identical state transitions for equivalent messages.**

### Task 5: Model capability mapping and deterministic seed

**Files:**
- Create: `tinfer/tinfer/server/websocket/capability_mapper.py`
- Modify: `tinfer/tinfer/models/impl/styletts2/model/inference_config.py`
- Create: `tinfer/tinfer/models/impl/styletts2/model/random_state.py`
- Create: `tinfer/tinfer/models/impl/styletts2/model/batch_text.py`
- Create: `tinfer/tinfer/models/impl/styletts2/model/batch_style.py`
- Create: `tinfer/tinfer/models/impl/styletts2/model/batch_audio.py`
- Modify: `tinfer/tinfer/models/impl/styletts2/model/model.py`
- Modify: `tinfer/tinfer/process/worker.py`
- Test: `tmp_tests/tts_api/test_capability_mapping.py`
- Test: `tmp_tests/styletts2/test_seeded_generation.py`

**Interfaces:**
- Produces: `CapabilityMapper.styletts2(settings: VoiceSettings, seed: int | None) -> dict[str, Any]`
- Produces: `request_generator(seed: int | None, device: torch.device) -> torch.Generator | None`
- Produces: `prepare_text_batch(model, texts, configs, alignment_type) -> PreparedModelBatch`
- Produces: `prepare_style_batch(model, prepared, references, previous, configs, generators) -> StyleBatch`
- Produces: `decode_audio_batch(model, prepared, styles, configs) -> list[ModelResult]`

- [ ] **Step 1: Add failing pure mapping tests and a seeded-generation test proving equal text, voice, parameters, and seed produce equal audio.**

```python
mapped = CapabilityMapper.styletts2(VoiceSettings(stability=0.8, similarity_boost=0.75, style=0.4, speed=1.1), 7)
self.assertEqual(mapped["speed"], 1.1)
self.assertEqual(mapped["style_interpolation_factor"], 0.8)
self.assertEqual(mapped["alpha"], 0.25)
self.assertEqual(mapped["beta"], 0.25)
self.assertEqual(mapped["embedding_scale"], 1.4)
```

- [ ] **Step 2: Run the mapping and seeded tests; expect missing mappings and nondeterministic diffusion.**

- [ ] **Step 3: Implement clamped mappings, make explicit `alpha`/`beta` override derived values, and pass a request-local generator into diffusion sampling without changing global RNG state.**

```python
def request_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
```

For runtimes that cannot accept `embedding_scale` or a generator, omit only that mapped key and record the field as accepted/no-effect for that runtime.

- [ ] **Step 4: Move text preparation, style sampling/blending, and predictor/decoder execution into `batch_text.py`, `batch_style.py`, and `batch_audio.py`. Keep `StyleTTS2.generate_batch()` as orchestration over the three typed results and keep every touched model file below 300 lines.**

- [ ] **Step 5: Run StyleTTS2, capability, TensorRT, and live synthesis tests; expect deterministic seeded PyTorch output and documented no-effect runtime exceptions.**

---

Continue with `docs/superpowers/plans/2026-07-13-speech-api-drop-in-media-and-discovery.md` after Task 5 passes.
