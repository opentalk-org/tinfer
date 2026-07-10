const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tinfer", {
  syncCatalog: (target) => ipcRenderer.invoke("catalog:sync", target),
  startSynthesis: (request) => ipcRenderer.invoke("synthesis:start", request),
  stopSynthesis: () => ipcRenderer.invoke("synthesis:stop"),
  sendChunk: (text) => ipcRenderer.invoke("synthesis:chunk", text),
  forceSynthesis: () => ipcRenderer.invoke("synthesis:force"),
  endStream: () => ipcRenderer.invoke("synthesis:end"),
  onSynthesisEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("synthesis:event", listener);
    return () => ipcRenderer.removeListener("synthesis:event", listener);
  },
});
