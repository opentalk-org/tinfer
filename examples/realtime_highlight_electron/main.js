const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("node:path");

const apiClient = require("./api-client");
const grpcClient = require("./grpc-client");
const { audioEvent, clearRun, createRun, currentRun } = require("./synthesis-run");

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 720,
    minHeight: 560,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

function sendEvent(payload) {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send("synthesis:event", payload);
}

function startSynthesis(request) {
  const run = createRun();
  const startedAt = performance.now();
  const client = request.protocol === "grpc" ? grpcClient : apiClient;
  sendEvent({ type: "status", status: `Running ${request.mode}` });
  return new Promise((resolve) => {
    let finished = false;
    const emit = (audioBase64, alignments) => {
      if (!run.cancelled && audioBase64) sendEvent(audioEvent(run, startedAt, audioBase64, alignments));
    };
    const finish = (error) => {
      if (finished) return;
      finished = true;
      if (error && !run.cancelled) sendEvent({ type: "error", message: error.message || String(error) });
      else if (!run.cancelled) sendEvent({ type: "done" });
      clearRun(run);
      resolve({ ok: !error || run.cancelled, cancelled: run.cancelled });
    };
    client.start(request, run, emit, finish);
  });
}

ipcMain.handle("catalog:sync", (_event, target) => {
  if (target.protocol === "grpc") {
    return grpcClient.fetchCatalog(apiClient.normalizeHostPort(target.host, target.port));
  }
  return apiClient.fetchCatalog(target.host, target.port);
});

ipcMain.handle("synthesis:start", async (_event, request) => {
  if (!request || !String(request.text).trim()) return { ok: false, error: "Text is empty." };
  if (request.protocol === "grpc") {
    request.address = apiClient.normalizeHostPort(request.host, request.port);
  }
  try {
    return await startSynthesis(request);
  } catch (error) {
    sendEvent({ type: "error", message: error.message || String(error) });
    return { ok: false, error: error.message || String(error) };
  }
});

ipcMain.handle("synthesis:stop", () => {
  const run = currentRun();
  if (run) run.cancel();
  sendEvent({ type: "stopped" });
  return { ok: true };
});

ipcMain.handle("synthesis:chunk", (_event, text) => {
  const run = currentRun();
  if (run) run.sendChunk(text);
  return { ok: true };
});

ipcMain.handle("synthesis:force", () => {
  const run = currentRun();
  if (run) run.force();
  return { ok: true };
});

ipcMain.handle("synthesis:end", () => {
  const run = currentRun();
  if (run) run.end();
  return { ok: true };
});

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  const run = currentRun();
  if (run) run.cancel();
  if (process.platform !== "darwin") app.quit();
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
