# Client examples

- [Python](python/README.md): HTTP, WebSocket, and gRPC clients, alignment, latency, and parameter comparisons.
- [Browser](browser/): open `index.html` to try streaming synthesis in a browser.
- [Electron](electron/README.md): desktop playback with word highlighting.

All clients connect to the Rust server. The Python and Electron gRPC clients use
the service contract in `tinfer_rust/proto/styletts.proto`.
