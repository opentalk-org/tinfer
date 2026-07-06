const protocolEl = document.getElementById("protocol");
const modeEl = document.getElementById("mode");
const hostEl = document.getElementById("host");
const portEl = document.getElementById("port");
const modelEl = document.getElementById("model");
const voiceEl = document.getElementById("voice");
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

// Small scheduling lookahead so a chunk is queued slightly ahead of the clock.
const LOOKAHEAD_S = 0.03;

const MODE_OPTIONS = {
  grpc: [
    { value: "unary", label: "Unary" },
    { value: "stream", label: "Stream" },
    { value: "incremental", label: "Incremental" },
  ],
  websocket: [{ value: "streaming", label: "Streaming" }],
};
const DEFAULT_MODE = { grpc: "stream", websocket: "streaming" };
const DEFAULT_SCHEDULE = [120, 160, 250, 290];

// Parse the comma-separated chunk schedule; fall back to the default when the
// field is blank or garbage so a stray keystroke can't send an empty schedule.
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

function setStatus(value, isError = false) {
  statusEl.textContent = value;
  statusEl.classList.toggle("error", isError);
}

// Elapsed playback time on the AudioContext clock. Because chunks are scheduled
// on this same clock, underrun gaps are baked into both audio and timeline —
// so highlight and progress bars stay in sync with what you actually hear.
function elapsedMs() {
  if (!state.audioContext || state.audioStartTime == null) return 0;
  return (state.audioContext.currentTime - state.audioStartTime) * 1000;
}

function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function stopSources() {
  for (const source of state.sources) {
    try { source.stop(); } catch (_) {}
  }
  state.sources = [];
}

function clearTimeline() {
  for (const seg of state.segments) seg.el.remove();
  state.segments = [];
  trackEl.classList.remove("has-chunks");
}

function resetPlayback() {
  stopSources();
  stopTimer();
  state.audioStartTime = null;
  state.nextStartTime = 0;
  state.timeline = [];
  state.alignCursor = 0;
  state.lastSpanIndex = -1;
  state.pcmChunks = [];
  state.done = false;
  saveEl.disabled = true;
  clearTimeline();
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
}

// Wrap concatenated 16-bit mono PCM chunks in a 44-byte WAV/RIFF header.
function pcmChunksToWavBlob(chunks, sampleRate) {
  let dataLen = 0;
  for (const c of chunks) dataLen += c.length;
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 2 bytes/sample)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLen, true);
  return new Blob([header, ...chunks], { type: "audio/wav" });
}

function saveWav() {
  if (!state.pcmChunks.length) return;
  const blob = pcmChunksToWavBlob(state.pcmChunks, state.sampleRate);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `tinfer-${Date.now()}.wav`;
  link.click();
  URL.revokeObjectURL(url);
}

function base64ToBytes(base64) {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

function pcmToAudioBuffer(base64, sampleRate) {
  const bytes = base64ToBytes(base64);
  const samples = Math.floor(bytes.length / 2);
  if (!samples) return null;
  const buffer = state.audioContext.createBuffer(1, samples, sampleRate);
  const channel = buffer.getChannelData(0);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples; i += 1) {
    channel[i] = view.getInt16(i * 2, true) / 32768;
  }
  return buffer;
}

function splitWords(text) {
  const parts = [];
  const pattern = /\S+|\s+/g;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    parts.push({ text: match[0], word: !/^\s+$/.test(match[0]) });
  }
  return parts;
}

// Array.from keeps multi-byte glyphs (Polish ą/ś/ż) as single tokens so char
// spans line up 1:1 with the WebSocket char alignment stream.
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

// gRPC alignment tokens tile the sent text exactly, but the server splits
// punctuation into its own token while splitWords glues it to the word — so a
// token-index match drifts one step per punctuation mark. Map by character
// offset instead: validate each token against the text it must tile, then
// attribute it to the word span holding its first letter.
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

// Word a token belongs to: the span holding its first letter. Pure
// punctuation/whitespace tokens (a pause) inherit the previous word so the
// highlight lingers on it rather than blanking mid-sentence.
function ownerSpanIndex(charStart, word) {
  const rel = word.search(/\p{L}/u);
  if (rel >= 0) {
    const at = charStart + rel;
    const idx = state.wordSpans.findIndex((s) => at >= s.charStart && at < s.charEnd);
    if (idx >= 0) state.lastSpanIndex = idx;
  }
  return state.lastSpanIndex;
}

function updateSpokenSpansByChar(elapsed) {
  let spokenMax = -1;
  let current = -1;
  for (const item of state.timeline) {
    if (elapsed >= item.endMs) {
      spokenMax = Math.max(spokenMax, item.spanIndex);
    } else if (elapsed >= item.startMs) {
      current = item.spanIndex;
      break;
    } else {
      break;
    }
  }
  if (current < 0) current = spokenMax + 1;
  state.wordSpans.forEach((word, index) => {
    word.el.classList.toggle("spoken", index < current);
    word.el.classList.toggle("current", index === current);
  });
}

// WebSocket char/word alignment drops whitespace symmetrically on both the span
// and timeline side, so token index i maps cleanly to word span i.
function updateSpokenSpansByIndex(elapsed) {
  let currentIndex = -1;
  let spokenCount = 0;
  for (let i = 0; i < state.timeline.length; i += 1) {
    const item = state.timeline[i];
    if (elapsed >= item.endMs) {
      spokenCount = i + 1;
    } else if (elapsed >= item.startMs) {
      currentIndex = i;
      break;
    }
  }
  if (currentIndex < 0 && spokenCount < state.timeline.length) currentIndex = spokenCount;
  state.wordSpans.forEach((word, index) => {
    word.el.classList.toggle("spoken", index < spokenCount);
    word.el.classList.toggle("current", index === currentIndex);
  });
}

function updateSpokenSpans(elapsed) {
  if (state.protocol === "grpc") updateSpokenSpansByChar(elapsed);
  else updateSpokenSpansByIndex(elapsed);
}

// Each chunk is a card sized to its text; the bar underneath fills as the chunk
// plays and the card lights up while it is the one being spoken.
function addSegment(payload, startMs, endMs, durationMs) {
  trackEl.classList.add("has-chunks");

  const el = document.createElement("div");
  el.className = "segment";

  const tokens = document.createElement("div");
  tokens.className = "seg-tokens";
  const words = payload.alignments || [];
  if (words.length) {
    for (const item of words) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = item.word;
      tokens.appendChild(chip);
    }
  } else {
    const chip = document.createElement("span");
    chip.className = "chip chip-muted";
    chip.textContent = "♪ audio";
    tokens.appendChild(chip);
  }

  const meta = document.createElement("div");
  meta.className = "seg-meta";
  const ttfb = payload.firstByteMs != null ? ` · ${payload.firstByteMs} ms TTFB` : "";
  meta.textContent = `#${state.segments.length} · ${Math.round(durationMs)} ms${ttfb}`;

  const bar = document.createElement("div");
  bar.className = "seg-bar";
  const fill = document.createElement("div");
  fill.className = "seg-fill";
  bar.appendChild(fill);

  el.append(tokens, meta, bar);
  trackEl.appendChild(el);
  state.segments.push({ el, fill, startMs, endMs, durationMs });
  trackEl.scrollLeft = trackEl.scrollWidth;
}

function updateProgress(elapsed) {
  for (const seg of state.segments) {
    const ratio = seg.durationMs > 0
      ? (elapsed - seg.startMs) / seg.durationMs
      : (elapsed >= seg.endMs ? 1 : 0);
    const clamped = Math.max(0, Math.min(1, ratio));
    seg.fill.style.width = `${(clamped * 100).toFixed(1)}%`;
    seg.el.classList.toggle("current", elapsed >= seg.startMs && elapsed < seg.endMs);
    seg.el.classList.toggle("played", elapsed >= seg.endMs);
  }
}

// On cancel, keep chunks already started and fade out the ones still pending.
function cancelTimeline(elapsed) {
  const kept = [];
  for (const seg of state.segments) {
    if (seg.startMs >= elapsed) {
      const el = seg.el;
      el.classList.add("fading");
      setTimeout(() => el.remove(), 260);
    } else {
      kept.push(seg);
    }
  }
  state.segments = kept;
}

function finishIfIdle() {
  if (!state.done) return;
  if (state.audioContext && state.audioStartTime != null && state.audioContext.currentTime < state.nextStartTime) return;
  stopTimer();
  state.live = false;
  updateControls();
  setStatus("Idle");
}

function updateHighlight() {
  const elapsed = elapsedMs();
  updateSpokenSpans(elapsed);
  updateProgress(elapsed);
  finishIfIdle();
}

async function enqueueAudio(payload) {
  if (!state.audioContext) state.audioContext = new AudioContext();
  const ctx = state.audioContext;
  if (ctx.state === "suspended") await ctx.resume();

  const buffer = pcmToAudioBuffer(payload.audioBase64, payload.sampleRate);
  if (!buffer) return;
  const durationMs = payload.durationMs || buffer.duration * 1000;

  state.pcmChunks.push(base64ToBytes(payload.audioBase64));
  state.sampleRate = payload.sampleRate;
  saveEl.disabled = false;

  // Schedule seamlessly after the previous chunk; if we underran, start slightly
  // ahead of now and let the real gap show up in the timeline.
  const startAt = Math.max(ctx.currentTime + LOOKAHEAD_S, state.nextStartTime);
  if (state.audioStartTime == null) state.audioStartTime = startAt;
  const startMs = (startAt - state.audioStartTime) * 1000;
  const endMs = startMs + durationMs;
  state.nextStartTime = startAt + durationMs / 1000;

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(startAt);
  state.sources.push(source);

  try {
    for (const item of payload.alignments || []) {
      const entry = { word: item.word, startMs: startMs + item.startMs, endMs: startMs + item.endMs };
      if (state.protocol === "grpc") assignCharRange(entry, item.word);
      state.timeline.push(entry);
    }
  } catch (err) {
    setStatus(err.message, true);
    throw err;
  }
  addSegment(payload, startMs, endMs, durationMs);
  if (!state.timer) state.timer = setInterval(updateHighlight, 30);
}

function populateCatalog(catalog) {
  state.catalog = catalog.length ? catalog : [{ id: "magda", voices: ["magda_001"] }];
  modelEl.textContent = "";
  for (const model of state.catalog) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.id;
    modelEl.appendChild(option);
  }
  if (state.catalog.some((model) => model.id === "magda")) modelEl.value = "magda";
  populateVoices();
}

function populateVoices() {
  const model = state.catalog.find((item) => item.id === modelEl.value) || state.catalog[0];
  voiceEl.textContent = "";
  for (const voice of ["auto", ...model.voices]) {
    const option = document.createElement("option");
    option.value = voice;
    option.textContent = voice;
    voiceEl.appendChild(option);
  }
  if (model.voices.includes("magda_001")) voiceEl.value = "magda_001";
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

// gRPC Incremental and WS Streaming accept live text chunks / force / end mid-run.
function isIncrementalMode() {
  return (protocolEl.value === "grpc" && modeEl.value === "incremental") ||
    (protocolEl.value === "websocket" && modeEl.value === "streaming");
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
  if (protocolEl.value === "grpc" && portEl.value === "8001") portEl.value = "50051";
  if (protocolEl.value === "websocket" && portEl.value === "50051") portEl.value = "8001";
  // gRPC only carries word-level timings, so char highlighting is WebSocket-only.
  granularityEl.disabled = protocolEl.value === "grpc";
  // Alpha/beta/speed ride in WebSocket voice_settings; the gRPC config has no field for them.
  alphaEl.disabled = betaEl.disabled = speedEl.disabled = scheduleEl.disabled = protocolEl.value === "grpc";
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

modelEl.addEventListener("change", populateVoices);
protocolEl.addEventListener("change", () => {
  syncProtocolDefaults();
  populateModes();
  updateControls();
});
modeEl.addEventListener("change", updateControls);
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

window.tinfer.getCatalog().then(populateCatalog).catch(() => {
  populateCatalog([{ id: "magda", voices: ["magda_001"] }]);
});
