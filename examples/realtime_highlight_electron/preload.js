const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tinfer", {
  getCatalog: () => ipcRenderer.invoke("catalog:get"),
  startSynthesis: (request) => ipcRenderer.invoke("synthesis:start", request),
  stopSynthesis: () => ipcRenderer.invoke("synthesis:stop"),
  onSynthesisEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("synthesis:event", listener);
    return () => ipcRenderer.removeListener("synthesis:event", listener);
  },
});
