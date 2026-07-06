const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const WebSocket = require("ws");

const SAMPLE_RATE = 24000;
const OUTPUT_FORMAT = "pcm_24000";
const CHUNK_SCHEDULE = [120, 160, 250, 290];
const ROOT_DIR = path.resolve(__dirname, "../../..");
const CONVERTED_MODELS_DIR = path.join(ROOT_DIR, "converted_models");

const FALLBACK_CATALOG = [
  { id: "agnieszka", voices: ["agnieszka-best"] },
  { id: "magda", voices: ["magda_001"] },
  { id: "olam", voices: ["ola"] },
];

const protoDefinition = protoLoader.loadSync(path.join(__dirname, "styletts.proto"), {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});
const styletts = grpc.loadPackageDefinition(protoDefinition).styletts.v1;

let mainWindow = null;
let activeRun = null;

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
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send("synthesis:event", payload);
}

function readModelCatalog() {
  try {
    return fs.readdirSync(CONVERTED_MODELS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => {
        const modelId = entry.name;
        const voicesDir = path.join(CONVERTED_MODELS_DIR, modelId, "voices");
        let voices = [];
        try {
          voices = fs.readdirSync(voicesDir)
            .filter((name) => name.endsWith(".pth"))
            .map((name) => path.basename(name, ".pth"))
            .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        } catch (_) {
          voices = [];
        }
        return { id: modelId, voices };
      })
      .filter((model) => model.voices.length > 0)
      .sort((a, b) => a.id.localeCompare(b.id));
  } catch (_) {
    return FALLBACK_CATALOG;
  }
}

function normalizeHostPort(host, port) {
  const cleanHost = String(host || "localhost")
    .replace(/^https?:\/\//, "")
    .replace(/^wss?:\/\//, "")
    .replace(/\/.*$/, "")
    .trim() || "localhost";
  const cleanPort = String(port || "").trim();
  if (cleanHost.includes(":") || !cleanPort) return cleanHost;
  return `${cleanHost}:${cleanPort}`;
}

function wsUrl(host, port, modelId, voiceId) {
  const target = normalizeHostPort(host, port);
  const params = new URLSearchParams({
    model_id: modelId,
    output_format: OUTPUT_FORMAT,
    sync_alignment: "true",
  });
  return `ws://${target}/v1/text-to-speech/${encodeURIComponent(voiceId)}/stream-input?${params}`;
}

function audioDurationMs(base64Audio) {
  return (Buffer.from(base64Audio, "base64").length / 2 / SAMPLE_RATE) * 1000;
}

function grpcAlignments(response) {
  const items = Array.isArray(response.alignments) ? response.alignments : [];
  return items
    .map((item) => ({
      word: item.word || "",
      startMs: Number(item.start_ms || 0),
      endMs: Number(item.end_ms || 0),
    }))
    .filter((item) => item.word && item.endMs >= item.startMs);
}

function wordsFromCharAlignment(alignment) {
  if (!alignment || !Array.isArray(alignment.chars)) return [];
  const chars = alignment.chars.map((char) => String(char));
  const starts = alignment.charStartTimesMs || [];
  const durations = alignment.charDurationsMs || [];
  const words = [];
  let current = null;

  for (let i = 0; i < chars.length; i += 1) {
    const char = chars[i];
    const startMs = Number(starts[i] || 0);
    const endMs = startMs + Number(durations[i] || 0);
    if (/\s/.test(char)) {
      if (current) {
        words.push(current);
        current = null;
      }
      continue;
    }
    if (!current) {
      current = { word: char, startMs, endMs };
    } else {
      current.word += char;
      current.endMs = Math.max(current.endMs, endMs);
    }
  }

  if (current) words.push(current);
  return words.filter((item) => item.word && item.endMs >= item.startMs);
}

function charsFromCharAlignment(alignment) {
  if (!alignment || !Array.isArray(alignment.chars)) return [];
  const starts = alignment.charStartTimesMs || [];
  const durations = alignment.charDurationsMs || [];
  const items = [];
  for (let i = 0; i < alignment.chars.length; i += 1) {
    const char = String(alignment.chars[i]);
    if (/\s/.test(char)) continue;
    const startMs = Number(starts[i] || 0);
    items.push({ word: char, startMs, endMs: startMs + Number(durations[i] || 0) });
  }
  return items.filter((item) => item.endMs >= item.startMs);
}

// A run's live controls are protocol-specific; each starter overwrites the
// no-op defaults so the renderer can drive incremental/streaming sessions.
function makeRun() {
  const run = {
    cancelled: false,
    firstByteMs: null,
    sendChunk() {},
    force() {},
    end() {},
    cancel() {},
  };
  activeRun = run;
  return run;
}

function clearRun(run) {
  if (activeRun === run) activeRun = null;
}

function emitAudio(run, startedAt, audioBase64, alignments) {
  if (run.cancelled || !audioBase64) return;
  const firstByteMs = run.firstByteMs == null ? Math.round(performance.now() - startedAt) : null;
  if (firstByteMs != null) run.firstByteMs = firstByteMs;
  sendEvent({
    type: "audio",
    audioBase64,
    sampleRate: SAMPLE_RATE,
    durationMs: audioDurationMs(audioBase64),
    alignments,
    firstByteMs,
  });
}

function grpcConfig(request) {
  return { model_id: request.modelId, voice_id: request.voiceId, sample_rate_hz: SAMPLE_RATE };
}

function grpcReadSide(run, startedAt, resolve) {
  return {
    data(response) {
      const audioBase64 = Buffer.from(response.audio_data || []).toString("base64");
      emitAudio(run, startedAt, audioBase64, grpcAlignments(response));
    },
    error(error) {
      if (!run.cancelled) sendEvent({ type: "error", message: error.details || error.message || String(error) });
      clearRun(run);
      resolve({ ok: run.cancelled, cancelled: run.cancelled });
    },
    end() {
      if (!run.cancelled) sendEvent({ type: "done" });
      clearRun(run);
      resolve({ ok: true });
    },
  };
}

function startGrpcUnary(client, request, run, startedAt) {
  sendEvent({ type: "status", status: "gRPC Synthesize (unary)" });
  return new Promise((resolve) => {
    const call = client.Synthesize({ text: request.text, config: grpcConfig(request) }, (error, response) => {
      if (error) {
        if (!run.cancelled) sendEvent({ type: "error", message: error.details || error.message || String(error) });
        clearRun(run);
        resolve({ ok: run.cancelled, cancelled: run.cancelled });
        return;
      }
      const audioBase64 = Buffer.from(response.audio_data || []).toString("base64");
      emitAudio(run, startedAt, audioBase64, grpcAlignments(response));
      if (!run.cancelled) sendEvent({ type: "done" });
      clearRun(run);
      resolve({ ok: true });
    });
    run.cancel = () => { run.cancelled = true; call.cancel(); };
  });
}

function startGrpcStream(client, request, run, startedAt) {
  sendEvent({ type: "status", status: "gRPC SynthesizeStream" });
  return new Promise((resolve) => {
    const stream = client.SynthesizeStream({ text: request.text, config: grpcConfig(request) });
    run.cancel = () => { run.cancelled = true; stream.cancel(); };
    const side = grpcReadSide(run, startedAt, resolve);
    stream.on("data", side.data);
    stream.on("error", side.error);
    stream.on("end", side.end);
  });
}

function startGrpcIncremental(client, request, run, startedAt) {
  sendEvent({ type: "status", status: "gRPC Incremental — send chunks, Force to flush, End to finish" });
  const call = client.SynthesizeIncremental();
  call.write({ config: grpcConfig(request) });
  if (request.text && request.text.trim()) call.write({ text_chunk: request.text });
  run.sendChunk = (text) => { if (text && text.trim()) call.write({ text_chunk: text }); };
  run.force = () => call.write({ force_synthesis: {} });
  run.end = () => call.end();
  run.cancel = () => {
    run.cancelled = true;
    try { call.write({ cancel: {} }); } catch (_) {}
    call.cancel();
  };
  return new Promise((resolve) => {
    const side = grpcReadSide(run, startedAt, resolve);
    call.on("data", side.data);
    call.on("error", side.error);
    call.on("end", side.end);
  });
}

// Unary returns the whole utterance in one message; the client is the receiver
// and its default 4 MB cap rejects long audio ("Received message larger than max").
const GRPC_CHANNEL_OPTIONS = {
  "grpc.max_receive_message_length": 64 * 1024 * 1024,
  "grpc.max_send_message_length": 64 * 1024 * 1024,
};

async function startGrpc(request) {
  const run = makeRun();
  const startedAt = performance.now();
  const address = normalizeHostPort(request.host, request.port);
  const client = new styletts.StyleTTSService(address, grpc.credentials.createInsecure(), GRPC_CHANNEL_OPTIONS);
  if (request.mode === "unary") return startGrpcUnary(client, request, run, startedAt);
  if (request.mode === "incremental") return startGrpcIncremental(client, request, run, startedAt);
  return startGrpcStream(client, request, run, startedAt);
}

async function startWebSocket(request) {
  const run = makeRun();
  const startedAt = performance.now();
  const url = wsUrl(request.host, request.port, request.modelId, request.voiceId);

  sendEvent({ type: "status", status: "Connecting to WebSocket" });

  return new Promise((resolve) => {
    const ws = new WebSocket(url, { maxPayload: 10 * 1024 * 1024 });
    run.cancel = () => { run.cancelled = true; if (ws.readyState <= WebSocket.OPEN) ws.close(); };
    run.sendChunk = (text) => { if (ws.readyState === WebSocket.OPEN && text && text.trim()) ws.send(JSON.stringify({ text })); };
    run.force = () => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ text: "", try_trigger_generation: true })); };
    run.end = () => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ text: "" })); };

    ws.on("open", () => {
      if (run.cancelled) return;
      sendEvent({ type: "status", status: "WebSocket streaming — Send chunk, Force to flush, End to finish" });
      ws.send(JSON.stringify({
        text: request.text,
        voice_settings: { alpha: request.alpha, beta: request.beta, speed: request.speed },
        generation_config: { chunk_length_schedule: request.chunkSchedule || CHUNK_SCHEDULE },
      }));
    });

    ws.on("message", (message) => {
      if (run.cancelled) return;
      let data;
      try {
        data = JSON.parse(String(message));
      } catch (_) {
        return;
      }
      if (data.error) {
        sendEvent({ type: "error", message: data.error });
        ws.close();
        return;
      }
      if (data.isFinal) {
        sendEvent({ type: "done" });
        ws.close();
        return;
      }
      if (data.audio) {
        const alignment = data.normalizedAlignment || data.alignment;
        const toAlignments = request.granularity === "char" ? charsFromCharAlignment : wordsFromCharAlignment;
        emitAudio(run, startedAt, data.audio, toAlignments(alignment));
      }
    });

    ws.on("error", (error) => {
      if (!run.cancelled) sendEvent({ type: "error", message: error.message || String(error) });
    });

    ws.on("close", () => {
      clearRun(run);
      resolve({ ok: run.cancelled, cancelled: run.cancelled });
    });
  });
}

ipcMain.handle("catalog:get", () => readModelCatalog());

ipcMain.handle("synthesis:stop", () => {
  if (activeRun) activeRun.cancel();
  sendEvent({ type: "stopped" });
  return { ok: true };
});

ipcMain.handle("synthesis:chunk", (_event, text) => {
  if (activeRun) activeRun.sendChunk(text);
  return { ok: true };
});

ipcMain.handle("synthesis:force", () => {
  if (activeRun) activeRun.force();
  return { ok: true };
});

ipcMain.handle("synthesis:end", () => {
  if (activeRun) activeRun.end();
  return { ok: true };
});

ipcMain.handle("synthesis:start", async (_event, request) => {
  if (activeRun) activeRun.cancel();
  if (!request || !String(request.text || "").trim()) {
    return { ok: false, error: "Text is empty." };
  }
  try {
    if (request.protocol === "websocket") {
      return await startWebSocket(request);
    }
    return await startGrpc(request);
  } catch (error) {
    sendEvent({ type: "error", message: error.message || String(error) });
    return { ok: false, error: error.message || String(error) };
  }
});

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (activeRun) activeRun.cancel();
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
