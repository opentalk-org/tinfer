## Tinfer StyleTTS 2.1
tinfer: Fast production ready TTS inference server with gRPC and elevenlabs-compatible APIs.

## Features

- High-performance and ultra low latency Text To Speech model.
- Ready to use OCI container, fast deployment.
- Optimized for low latency and high througput.
- Websocket (elevenlabs compatible) and GRPC streaming api.
- Modular, easy to add new models in the future.

## Development (Nix)

The devshell and the serving image share one set of dependency lists in
`flake.nix` and install Python packages from the same `uv.lock`, so the dev
environment can't drift from what ships.

```bash
nix develop            # or `direnv allow` once
uv sync --all-packages # whole workspace; client-only: uv sync --package tinfer
python -m server.main --smoke-test
python -m server.main  # needs converted_models/
```

`.venv` is built against the image's nix interpreter and auto-activated on
shell entry. GPU works on bare hosts: the shell adds the host NVIDIA driver
dirs to `LD_LIBRARY_PATH` and probes `TRITON_LIBCUDA_PATH`.

## Prerequirements (non-Nix)

- [espeak-ng](https://github.com/espeak-ng/espeak-ng): Required for phonemizer functionality. On Debian/Ubuntu, install `espeak-ng` and `libespeak-ng-dev` so Rust builds can link `libespeak-ng`.
- Python >= 3.12

## Installation

- **Client only**: `pip install .`
- **Server / local inference**: `pip install .[inference]`
