## Tinfer StyleTTS 2.1
tinfer: Fast production ready TTS inference server with gRPC and elevenlabs-compatible APIs.

## Features

- High-performance and ultra low latency Text To Speech model.
- Ready to use OCI container, fast deployment.
- Optimized for low latency and high througput.
- Websocket (elevenlabs compatible) and GRPC streaming api.
- Modular, easy to add new models in the future.

## Prerequirements

## Prerequirements

- [espeak-ng](https://github.com/espeak-ng/espeak-ng): Required for phonemizer functionality.
- Python >= 3.11

## Installation

- **Client only**: `pip install .`
- **Server / local inference**: `pip install .[inference]`
