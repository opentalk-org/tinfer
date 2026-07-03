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
shell entry. Native deps of the wheels (libstdc++, zlib, espeak-ng, the host
NVIDIA driver) are preloaded by absolute path via a generated
`sitecustomize.py` — no `LD_LIBRARY_PATH`, so other programs in the shell
are unaffected.

## Prerequirements (non-Nix)

- [espeak-ng](https://github.com/espeak-ng/espeak-ng): Required for phonemizer functionality (linked by the nix-built espeak_align module; the flake devShell provides it).
- Python >= 3.12

## Installation

- **Client only**: `pip install .`
- **Server / local inference**: `pip install .[inference]`

## VastAI installation

```bash
apt-get update
apt-get install -y espeak-ng libespeak-ng-dev

uv sync --package tinfer --extra inference
```
