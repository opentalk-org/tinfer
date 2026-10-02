const protocolEl = document.getElementById("protocol");
const modeEl = document.getElementById("mode");
const hostEl = document.getElementById("host");
const portEl = document.getElementById("port");
const modelEl = document.getElementById("model");
const voiceEl = document.getElementById("voice");
const syncEl = document.getElementById("sync");
const languageEl = document.getElementById("language");
const granularityEl = document.getElementById("granularity");
const alphaEl = document.getElementById("alpha");
const betaEl = document.getElementById("beta");
const speedEl = document.getElementById("speed");
const scheduleEl = document.getElementById("schedule");
const textEl = document.getElementById("text");
const sendEl = document.getElementById("send");
const cancelEl = document.getElementById("cancel");
const saveEl = document.getElementById("save");
const chunkEl = document.getElementById("chunk");
const sendChunkEl = document.getElementById("send-chunk");
const forceEl = document.getElementById("force");
const endEl = document.getElementById("end");
const highlightEl = document.getElementById("highlight");
const firstByteEl = document.getElementById("first-byte");
const statusEl = document.getElementById("status");
const trackEl = document.getElementById("track");

const LOOKAHEAD_S = 0.03;

const MODE_OPTIONS = {
  grpc: [
    { value: "unary", label: "Unary" },
    { value: "stream", label: "Stream" },
    { value: "incremental", label: "Incremental" },
  ],
  api: [
    { value: "ws_single", label: "WebSocket — single context" },
    { value: "ws_multi", label: "WebSocket — multiple contexts" },
    { value: "post_audio", label: "POST — audio" },
    { value: "post_timing", label: "POST — timing" },
    { value: "stream_audio", label: "Streaming POST — audio" },
    { value: "stream_timing", label: "Streaming POST — timing" },
  ],
};
const DEFAULT_MODE = { grpc: "stream", api: "ws_single" };
const DEFAULT_SCHEDULE = [120, 160, 250, 290];

function parseSchedule(value) {
  const nums = value.split(",").map((s) => Number(s.trim())).filter((n) => Number.isFinite(n) && n > 0);
  return nums.length ? nums : DEFAULT_SCHEDULE;
}

const state = {
  catalog: [],
  audioContext: null,
  audioStartTime: null, // AudioContext time (s) the first chunk begins playing
  nextStartTime: 0, // AudioContext time (s) the next chunk should begin
  sources: [], // scheduled BufferSources, so cancel can stop them
  timeline: [], // per-token {word, startMs, endMs, charStart?, charEnd?} on the playback clock
  wordSpans: [], // per highlighted word {el, charStart, charEnd}
  segments: [], // per-chunk cards {el, fill, startMs, endMs, durationMs}
  protocol: null, // "grpc" | "websocket" — selects the highlight mapping
  alignText: "", // text the gRPC alignment tokens must tile, for char-cursor mapping
  alignCursor: 0, // running character offset while consuming gRPC alignment tokens
  lastSpanIndex: -1, // last word a gRPC token mapped to, inherited by pause tokens
  pcmChunks: [], // raw Int16 PCM bytes per chunk, concatenated on Save WAV
  sampleRate: null, // sample rate of the accumulated PCM
  timer: null,
  done: false,
  live: false,
};

function splitWords(text) {
  const parts = [];
  const pattern = /\S+|\s+/g;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    parts.push({ text: match[0], word: !/^\s+$/.test(match[0]) });
  }
  return parts;
}

function splitChars(text) {
  return Array.from(text).map((ch) => ({ text: ch, word: !/\s/.test(ch) }));
}

function buildHighlight(text, granularity) {
  highlightEl.textContent = "";
  state.wordSpans = [];
  const tokens = granularity === "char" ? splitChars(text) : splitWords(text);
  let pos = 0;
  for (const part of tokens) {
    const span = document.createElement("span");
    span.textContent = part.text;
    const charStart = pos;
    pos += part.text.length;
    if (part.word) {
      span.className = "word";
      state.wordSpans.push({ el: span, charStart, charEnd: pos });
    }
    highlightEl.appendChild(span);
  }
}

function assignCharRange(entry, word) {
  const at = state.alignCursor;
  const slice = state.alignText.slice(at, at + word.length);
  if (slice !== word) {
    throw new Error(`alignment desync at char ${at}: sent ${JSON.stringify(slice)} != alignment ${JSON.stringify(word)}`);
  }
  entry.charStart = at;
  entry.charEnd = at + word.length;
  state.alignCursor = at + word.length;
  entry.spanIndex = ownerSpanIndex(at, word);
}

function ownerSpanIndex(charStart, word) {
  const rel = word.search(/\p{L}/u);
  if (rel >= 0) {
    const at = charStart + rel;
    const idx = state.wordSpans.findIndex((s) => at >= s.charStart && at < s.charEnd);
    if (idx >= 0) state.lastSpanIndex = idx;
  }
  return state.lastSpanIndex;
}

function populateCatalog(catalog) {
  state.catalog = catalog;
  window.TinferCatalogUi.populate(catalog, modelEl, voiceEl, languageEl);
}

function populateModes() {
  modeEl.textContent = "";
  for (const opt of MODE_OPTIONS[protocolEl.value]) {
    const option = document.createElement("option");
    option.value = opt.value;
    option.textContent = opt.label;
    modeEl.appendChild(option);
  }
  modeEl.value = DEFAULT_MODE[protocolEl.value];
}

function isIncrementalMode() {
  return (protocolEl.value === "grpc" && modeEl.value === "incremental") ||
    modeEl.value === "ws_single" || modeEl.value === "ws_multi";
}

function updateControls() {
  const streamLive = state.live && isIncrementalMode();
  sendEl.disabled = state.live;
  cancelEl.disabled = !state.live;
  chunkEl.disabled = !streamLive;
  sendChunkEl.disabled = !streamLive;
  forceEl.disabled = !streamLive;
  endEl.disabled = !streamLive;
}

function syncProtocolDefaults() {
  if (protocolEl.value === "grpc" && portEl.value === "8000") portEl.value = "50051";
  if (protocolEl.value === "api" && portEl.value === "50051") portEl.value = "8000";
  updateModeControls();
}

function updateModeControls() {
  const controls = window.TinferCatalogUi.controlState(protocolEl.value, modeEl.value);
  granularityEl.disabled = controls.granularityDisabled;
  alphaEl.disabled = betaEl.disabled = speedEl.disabled = controls.voiceSettingsDisabled;
  scheduleEl.disabled = controls.scheduleDisabled;
}

async function send() {
  resetPlayback();
  const text = textEl.value.trim();
  if (!text) {
    highlightEl.textContent = "";
    firstByteEl.textContent = "-";
    setStatus("Text is empty", true);
    return;
  }
  state.protocol = protocolEl.value;
  state.alignText = text;
  const granularity = protocolEl.value === "grpc" ? "word" : granularityEl.value;
  buildHighlight(text, granularity);
  firstByteEl.textContent = "-";
  state.live = true;
  updateControls();
  setStatus("Starting");
  await window.tinfer.startSynthesis({
    protocol: protocolEl.value,
    mode: modeEl.value,
    host: hostEl.value,
    port: portEl.value,
    modelId: modelEl.value,
    voiceId: voiceEl.value,
    language: languageEl.value,
    granularity,
    alpha: Number(alphaEl.value),
    beta: Number(betaEl.value),
    speed: Number(speedEl.value),
    chunkSchedule: parseSchedule(scheduleEl.value),
    text,
  });
}

async function cancel() {
  const elapsed = elapsedMs();
  stopSources();
  stopTimer();
  cancelTimeline(elapsed);
  updateProgress(elapsed);
  state.done = false;
  state.live = false;
  updateControls();
  await window.tinfer.stopSynthesis();
  setStatus("Stopped");
}

function sendChunk() {
  const text = chunkEl.value;
  if (!text.trim()) return;
  window.tinfer.sendChunk(text);
  chunkEl.value = "";
}

async function sync() {
  syncEl.disabled = true;
  setStatus("Syncing");
  try {
    const catalog = await window.tinfer.syncCatalog({ protocol: protocolEl.value, host: hostEl.value, port: portEl.value });
    populateCatalog(catalog);
    setStatus(`Synced ${catalog.length} models`);
  } catch (err) {
    setStatus(err.message || "Sync failed", true);
  } finally {
    syncEl.disabled = false;
  }
}

modelEl.addEventListener("change", () => window.TinferCatalogUi.updateSelection(state.catalog, modelEl, voiceEl, languageEl));
syncEl.addEventListener("click", sync);
protocolEl.addEventListener("change", () => {
  populateModes();
  syncProtocolDefaults();
  updateControls();
});
modeEl.addEventListener("change", () => {
  updateControls();
  updateModeControls();
});
sendEl.addEventListener("click", send);
cancelEl.addEventListener("click", cancel);
saveEl.addEventListener("click", saveWav);
sendChunkEl.addEventListener("click", sendChunk);
chunkEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendChunk();
});
forceEl.addEventListener("click", () => window.tinfer.forceSynthesis());
endEl.addEventListener("click", () => window.tinfer.endStream());

populateModes();
syncProtocolDefaults();
updateControls();

window.tinfer.onSynthesisEvent(async (event) => {
  if (event.type === "status") {
    setStatus(event.status);
    return;
  }
  if (event.type === "audio") {
    if (event.firstByteMs != null) firstByteEl.textContent = `${event.firstByteMs} ms`;
    setStatus("Playing");
    await enqueueAudio(event);
    return;
  }
  if (event.type === "done") {
    state.done = true;
    finishIfIdle();
    return;
  }
  if (event.type === "stopped") {
    stopSources();
    stopTimer();
    state.live = false;
    updateControls();
    setStatus("Stopped");
    return;
  }
  if (event.type === "error") {
    stopSources();
    stopTimer();
    state.live = false;
    updateControls();
    setStatus(event.message || "Error", true);
  }
});
