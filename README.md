## Tinfer StyleTTS 2.1
tinfer: Fast production ready TTS inference server with gRPC and elevenlabs-compatible APIs.

## Features

- High-performance and ultra low latency Text To Speech model.
- Ready to use OCI container, fast deployment.
- Optimized for low latency and high througput.
- Websocket (elevenlabs compatible) and GRPC streaming api.
- Modular, easy to add new models in the future.

## Development (Nix)

The flake ships a devshell that mirrors the serving image: both consume the
same dependency lists declared once in `flake.nix` (interpreter, native
toolchain, runtime libraries, env) and install Python packages from the same
`uv.lock`, so the dev environment can't drift from what ships.

```bash
nix develop            # or `direnv allow` once — .envrc uses the flake
uv sync --all-packages # install the whole workspace (server + tinfer[inference] + espeak_align)
python -m server.main --smoke-test  # verify runtime deps incl. CUDA end-to-end
python -m server.main               # start the server (needs converted_models/)
```

Notes:

- The devshell pins `UV_PYTHON` to the same nix `python311` the image is
  built with; `uv sync` creates `.venv` against it (re-entering the shell
  auto-activates `.venv`).
- GPU access works on bare hosts: the shell puts the host NVIDIA driver
  directories on `LD_LIBRARY_PATH` and points `TRITON_LIBCUDA_PATH` at the
  host `libcuda.so` (the image instead uses the nvidia container toolkit
  mount paths).
- Client-only work needs no extras: `uv sync --package tinfer` installs the
  lightweight client surface.

## Prerequirements (non-Nix)

- [espeak-ng](https://github.com/espeak-ng/espeak-ng): Required for phonemizer functionality. On Debian/Ubuntu, install `espeak-ng` and `libespeak-ng-dev` so Rust builds can link `libespeak-ng`.
- Python >= 3.12

## Installation

- **Client only**: `pip install .`
- **Server / local inference**: `pip install .[inference]`
