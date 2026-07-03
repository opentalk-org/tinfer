# Realtime highlight Electron example

Small Electron app for testing Tinfer streaming TTS. It can use either the gRPC
streaming API or the ElevenLabs-compatible WebSocket API, plays streamed PCM
audio, shows latency to the first audio byte, and highlights the current spoken
word.

## Run

```bash
cd /workspace/tinfer/examples/realtime_highlight_electron
. /opt/nvm/nvm.sh
npm install
npm start
```

Defaults:

- gRPC: `localhost:50051`
- WebSocket: `localhost:8000`
- sample rate/output: `pcm_24000`

The model/voice selectors are populated from `/workspace/converted_models`, with
fallbacks for the current converted model folders: `agnieszka`, `magda`, and
`olam`.
