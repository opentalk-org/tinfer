# Tinfer Rust Stage 5 Python Compatibility and Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `tinfer` wheel whose public interface matches the current Python package one-to-one, add operational lifecycle/telemetry, and prove the Rust runtime is ready for cutover.

**Architecture:** Maturin packages ordinary Python enums/dataclasses/facades plus one PyO3 extension linked to the same Rust library and unified C++ target as the executable. Python values are converted at the boundary; all scheduling and request-time model work stays Rust/C++. A generated manifest and dual-run black-box suite make compatibility measurable.

**Tech Stack:** PyO3, pyo3-async-runtimes, numpy, maturin, pytest, pytest-asyncio, aiohttp compatibility transport, tracing, OpenTelemetry.

## Global Constraints

- Apply every constraint in `2026-07-13-tinfer-rust-port.md`.
- The wheel starts no server executable, RPC sidecar, multiprocessing worker, or request-time Python inference callback.
- Preserve public imports, signatures/defaults, callable kinds, enums, dataclasses, NumPy dtypes/shapes, exceptions, generated protobuf modules, and lifecycle behavior.
- Private executor, worker, shared-memory, and scheduler internals are not compatibility surface.
- Build and test wheels outside the source tree so imports cannot accidentally resolve to `tinfer/tinfer/`.
- Run Python/maturin and Git commands from `/workspace/tinfer`; run Cargo commands from `/workspace/tinfer/tinfer_rust`.

---

### Task 1: Generate and Lock the Python API Manifest

**Files:**
- Create: `tinfer_rust/tools/api_manifest.py`
- Create: `tinfer_rust/tools/check_api_manifest.py`
- Create: `tinfer_rust/python_tests/fixtures/public_api.json`
- Test: `tinfer_rust/python_tests/test_api_manifest.py`

**Interfaces:**
- Produces: deterministic JSON records for modules, exports, signatures, callable kinds, enum members, dataclass fields/defaults, exceptions, and protobuf descriptors.

- [ ] **Step 1: Write the manifest checker test**

```python
def test_reference_manifest_is_stable(reference_package):
    actual = build_manifest(reference_package)
    expected = json.loads(API_FIXTURE.read_text())
    assert compatible_diff(expected, actual) == []
```

Inventory at least `config.engine_config`, `core.request`, `core.engine`, `core.async_engine`, `core.stream`, `server.health`, `server.grpc`, `server.grpc.server`, `server.websocket`, and `support`.

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tinfer_rust/python_tests/test_api_manifest.py -q`

Expected: import failure for `api_manifest`.

- [ ] **Step 3: Implement deterministic introspection**

Serialize `inspect.signature`, descriptor kind, `inspect.iscoroutinefunction`, async-generator status, enum name/value order, dataclass field order/default kind, `__all__`, exception bases, and protobuf serialized descriptors. Reject callable or opaque defaults with a clear manifest error instead of storing unstable `repr` output.

Generate the reviewed fixture from the current working-tree package in an isolated environment.

- [ ] **Step 4: Verify and commit**

Run: `python tinfer_rust/tools/api_manifest.py --package-root tinfer --output /tmp/api.json && diff -u tinfer_rust/python_tests/fixtures/public_api.json /tmp/api.json`

Expected: no diff.

```bash
git add tinfer_rust/tools tinfer_rust/python_tests
git commit -m "test(rust): lock current Python API manifest"
```

### Task 2: Package Value Types and Configuration

**Files:**
- Create: `tinfer_rust/pyproject.toml`
- Create: `tinfer_rust/python/tinfer/config/{__init__.py,engine_config.py}`
- Create: `tinfer_rust/python/tinfer/core/{__init__.py,request.py}`
- Create: `tinfer_rust/python/tinfer/support/{__init__.py,errors.py,observability.py}`
- Create: `tinfer_rust/src/python/{mod.rs,convert.rs,error.rs}`
- Test: `tinfer_rust/python_tests/test_values_config.py`

**Interfaces:**
- Produces: current `AlignmentType`, `StreamParams`, `ModelInfo`, `VoiceInfo`, `AlignmentItem`, `Alignment`, `AudioChunk`, `StreamingTTSConfig`, and `InferenceError` imports.

- [ ] **Step 1: Write exact value/config tests**

```python
def test_audio_chunk_contract():
    chunk = AudioChunk(np.zeros(4, np.float32), 24_000)
    assert dataclasses.fields(chunk)[0].name == "audio"
    assert chunk.chunk_index == 0
    assert chunk.text_span == (0, 0)

def test_config_round_trip(tmp_path):
    config = StreamingTTSConfig()
    config.to_yaml(tmp_path / "config.yml")
    assert StreamingTTSConfig.from_yaml(tmp_path / "config.yml") == config
```

- [ ] **Step 2: Confirm failure against an empty wheel**

Run: `maturin develop --manifest-path tinfer_rust/Cargo.toml && python -m pytest tinfer_rust/python_tests/test_values_config.py -q`

Expected: `ModuleNotFoundError` for compatibility modules.

- [ ] **Step 3: Implement ordinary Python values and native conversion**

Copy public definitions and exact defaults, not implementation modules. Keep dataclasses/enums/TypedDict in Python for correct introspection and pickling. Convert them immediately to typed Rust structs in `convert.rs`; convert Rust audio into contiguous one-dimensional NumPy arrays only after releasing native slots.

Map old configuration fields to automatic placement exactly as specified in the Python compatibility design. `executor_type="process"` remains accepted but creates no process; invalid backend/device pairs raise `ValueError`.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tinfer_rust/python_tests/test_values_config.py tinfer_rust/python_tests/test_api_manifest.py -q`

Expected: value/config tests pass; manifest differences remain only for engine/server modules not yet created.

```bash
git add tinfer_rust/pyproject.toml tinfer_rust/python tinfer_rust/src/python tinfer_rust/python_tests
git commit -m "feat(python): expose compatible values and configuration"
```

### Task 3: Engine, Stream, Async, and Registration Facades

**Files:**
- Create: `tinfer_rust/python/tinfer/core/{engine.py,async_engine.py,stream.py}`
- Create: `tinfer_rust/src/python/{engine.rs,stream.rs,asyncio.rs,model_import.rs}`
- Create: `tinfer_rust/src/models/stub/python_import.rs`
- Test: `tinfer_rust/python_tests/test_engine_stream.py`
- Test: `tinfer_rust/python_tests/test_registration.py`

**Interfaces:**
- Produces: `StreamingTTS`, `AsyncStreamingTTS`, `TTSStream`, and every current public method/signature.

- [ ] **Step 1: Write black-box facade tests**

```python
@pytest.mark.asyncio
async def test_async_generate_and_stream_lifecycle(stub_config):
    tts = AsyncStreamingTTS(stub_config)
    tts.load_model("stub", STUB_ARTIFACT)
    chunks = [chunk async for chunk in tts.generate("stub", "default", "hello", {})]
    assert chunks and all(chunk.audio.dtype == np.float32 for chunk in chunks)
    tts.stop()
```

Add exact tests for `from_config`, `generate_full`, `generate_full_batch`, catalogs, warmup, add/trigger/force/cancel, wait/pull/get/collect audio, state, repeated close/stop, event-loop affinity, cancellation, and current validation exception/message classes.

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tinfer_rust/python_tests/test_engine_stream.py -q`

Expected: missing `StreamingTTS` facade.

- [ ] **Step 3: Implement PyO3 handles and Python facades**

`PyEngine` holds `Arc<Engine>` and a Tokio handle. Synchronous waits call `py.detach`; async operations use pyo3 asyncio futures bound to the running loop. `TTSStream.pull_audio` is an ordinary Python async generator over native `_next_audio`; finalization sends best-effort cancellation but native correctness does not depend on GC.

Implement `register_model(model_id, model, device=None, keep_in_main=True)` by validating `_loaded`, calling `get_state()` once, and invoking a compile-time model-owned importer. The stub importer accepts the fixture's typed state and produces the same validated artifact data as `load_model`. Never retain the model PyObject in engine/native state. Apply current clearing behavior only after successful import when `keep_in_main` is false.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tinfer_rust/python_tests/test_engine_stream.py tinfer_rust/python_tests/test_registration.py -q`

Expected: all sync/async/registration contracts pass without a Python worker process.

```bash
git add tinfer_rust/python/tinfer/core tinfer_rust/src/python tinfer_rust/src/models/stub/python_import.rs tinfer_rust/python_tests
git commit -m "feat(python): expose compatible engine and streams"
```

### Task 4: Python Server and aiohttp Compatibility Facades

**Files:**
- Create: `tinfer_rust/python/tinfer/server/{__init__.py,health.py}`
- Create: `tinfer_rust/python/tinfer/server/grpc/{__init__.py,server.py}`
- Generate: `tinfer_rust/python/tinfer/server/grpc/{styletts_pb2.py,styletts_pb2_grpc.py}`
- Create: `tinfer_rust/python/tinfer/server/websocket/{__init__.py,server.py,handler.py}`
- Create: `tinfer_rust/src/python/{server.rs,http_adapter.rs,ws_adapter.rs}`
- Test: `tinfer_rust/python_tests/test_servers.py`
- Test: `tinfer_rust/python_tests/test_create_app.py`

**Interfaces:**
- Produces: `HealthState`, `GRPCServer`, `WebSocketServer`, `WebSocketHandler`, generated gRPC modules, and current coroutine lifecycle signatures.

- [ ] **Step 1: Write server and aiohttp adapter tests**

```python
@pytest.mark.asyncio
async def test_create_app_uses_rust_state_machine(async_tts):
    server = WebSocketServer(async_tts)
    app = server.create_app()
    client = await aiohttp_client(app)
    assert (await client.get("/health")).status == 200
    await assert_single_and_multi_websockets(client)
```

Also instantiate generated gRPC clients, exercise start/stop/serve, inspect signatures, test readiness/drain/admission, and repeat lifecycle calls according to the current contract.

- [ ] **Step 2: Confirm failure**

Run: `python -m pytest tinfer_rust/python_tests/test_servers.py tinfer_rust/python_tests/test_create_app.py -q`

Expected: missing server compatibility imports.

- [ ] **Step 3: Implement lifecycle and transport-only adapters**

Server facades own Rust server handles sharing the wrapped engine and health actor. Generate Python protobuf modules from the same proto during packaging. `create_app()` registers aiohttp handlers that pass parsed body/frame bytes into Rust HTTP services or Rust session handles and translate typed outputs back; Python owns only aiohttp request/socket transport. `WebSocketHandler` wraps the same single-session Rust handle. Do not duplicate validation or state transitions in Python.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tinfer_rust/python_tests/test_servers.py tinfer_rust/python_tests/test_create_app.py tmp_tests/tts_api -q`

Expected: Python lifecycle, aiohttp embedding, generated clients, and existing API tests pass against Rust.

```bash
git add tinfer_rust/python/tinfer/server tinfer_rust/src/python tinfer_rust/python_tests
git commit -m "feat(python): expose compatible Rust server facades"
```

### Task 5: Operations, Wheel Isolation, and Cutover Gate

**Files:**
- Create: `tinfer_rust/src/support/{telemetry.rs,shutdown.rs}`
- Create: `tinfer_rust/tests/shutdown.rs`
- Create: `tinfer_rust/python_tests/test_wheel_isolation.py`
- Create: `tinfer_rust/README.md`
- Create: `tinfer_rust/config/example.yaml`
- Modify: `.github/workflows/smoke-test.yml`
- Modify: `.github/workflows/docker.yml`

**Interfaces:**
- Produces: structured metrics/logs, deterministic shutdown ordering, standalone server configuration, and release gates.

- [ ] **Step 1: Write shutdown and isolated-wheel tests**

```rust
#[tokio::test]
async fn shutdown_destroys_in_dependency_order() {
    let events = run_instrumented_shutdown().await;
    assert_before(&events, "servers_stopped", "scheduler_stopped");
    assert_before(&events, "slots_destroyed", "primary_context_released");
}
```

Python isolation builds a wheel, installs it into a new virtual environment, removes the source repository from `PYTHONPATH`, imports every manifest module, runs one synthesis, then exits and asserts no child process remains.

- [ ] **Step 2: Confirm failure**

Run: `(cd tinfer_rust && cargo test --test shutdown) && python -m pytest tinfer_rust/python_tests/test_wheel_isolation.py -q`

Expected: missing instrumented shutdown and wheel release configuration.

- [ ] **Step 3: Implement operations and release configuration**

Emit queue depth, wait, batch size, slot utilization, inference/encoding latency, context retain identity, placement health, active transports, and error class with model/device/stage labels. Redact text and voice artifact contents. Shutdown drains admission, stops protocol accepts/sessions, stops scheduler/executors, destroys pipelines/slots, then releases primary contexts.

Document explicit and automatic placement YAML, backend Cargo features, library discovery, health semantics, Python wheel usage, and hardware test commands. CI has mandatory CPU jobs and separately labeled CUDA/TensorRT jobs.

- [ ] **Step 4: Run the full cutover gate and commit**

```bash
(cd tinfer_rust && cargo fmt --all -- --check)
(cd tinfer_rust && cargo clippy --workspace --all-targets --all-features -- -D warnings)
(cd tinfer_rust && cargo test --workspace --all-features)
python -m pytest tinfer_rust/python_tests tmp_tests/tts_api tmp_tests/styletts2 -q
python tinfer_rust/tools/check_api_manifest.py --reference tinfer --candidate tinfer-rust-wheel
```

Expected: every command exits 0, manifest reports no incompatible differences, isolated wheel uses no source package, and context metrics remain one retain per active GPU under stress.

```bash
git add tinfer_rust .github/workflows/smoke-test.yml .github/workflows/docker.yml
git commit -m "feat(rust): complete Python-compatible production cutover"
```
