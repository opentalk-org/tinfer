# Python clients for Tinfer Rust

Start the Rust server with a model and voice, then run clients from the repository root:

```bash
uv sync --group examples
uv run python examples/python/basic.py
uv run python examples/python/alignment.py
uv run python examples/python/websocket_client.py
uv run python examples/python/latencies.py
```

Set `TINFER_HTTP_URL`, `TINFER_GRPC_ADDRESS`, `TINFER_MODEL_ID`, and `TINFER_VOICE_ID`
to match the server. Defaults are localhost ports 8000/50051 and Magda/magda_001.

Generate gRPC clients from the Rust service contract before running gRPC examples:

```bash
uv run python examples/python/grpc_support/generate.py
uv run python examples/python/grpc_client.py
uv run python examples/python/grpc_alignment.py
uv run python examples/python/grpc_latencies.py
```

`multimodel.py` accepts two model/voice pairs. `grid_tts_params.py` compares settings
exposed by the HTTP API. WAV output is saved under `artifacts/outputs/examples/`.
