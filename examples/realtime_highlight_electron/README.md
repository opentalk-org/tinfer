# Realtime Highlight Electron Example

This app plays Tinfer PCM audio as it arrives and highlights available timing data.

## Run

Start Tinfer's gRPC server on port `50051` and HTTP/WebSocket server on port `8000`, then:

```bash
npm install
npm start
```

Sync the catalog after choosing a protocol. Model selection rebuilds the language list from the model metadata and selects its baked default.

The gRPC modes are unary, server streaming, and incremental. The API modes cover single-context WebSocket, multi-context WebSocket, regular audio/timing POST, and streaming audio/timing POST. Plain audio modes support playback and WAV saving without text highlighting.

## Verify

```bash
npm test
npm run check
```
