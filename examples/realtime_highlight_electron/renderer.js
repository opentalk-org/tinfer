const protocolEl = document.getElementById("protocol");
const hostEl = document.getElementById("host");
const portEl = document.getElementById("port");
const modelEl = document.getElementById("model");
const voiceEl = document.getElementById("voice");
const textEl = document.getElementById("text");
const sendEl = document.getElementById("send");
const stopEl = document.getElementById("stop");
const highlightEl = document.getElementById("highlight");
const firstByteEl = document.getElementById("first-byte");
const statusEl = document.getElementById("status");

const state = {
  catalog: [],
  audioContext: null,
  queue: [],
  currentSource: null,
  startedAtAudioTime: 0,
  timeline: [],
  wordSpans: [],
  timer: null,
  totalAudioMs: 0,
  done: false,
  playing: false,
};

function setStatus(value, isError = false) {
  statusEl.textContent = value;
  statusEl.classList.toggle("error", isError);
}

function resetPlayback() {
  if (state.currentSource) {
    try { state.currentSource.stop(); } catch (_) {}
  }
  if (state.timer) clearInterval(state.timer);
  state.queue = [];
  state.currentSource = null;
  state.startedAtAudioTime = 0;
  state.timeline = [];
  state.timer = null;
  state.totalAudioMs = 0;
  state.done = false;
  state.playing = false;
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

function buildHighlight(text) {
  highlightEl.textContent = "";
  state.wordSpans = [];
  for (const part of splitWords(text)) {
    const span = document.createElement("span");
    span.textContent = part.text;
    if (part.word) {
      span.className = "word";
      state.wordSpans.push(span);
    }
    highlightEl.appendChild(span);
  }
}

function updateHighlight() {
  if (!state.audioContext || !state.timeline.length) return;
  const elapsedMs = (state.audioContext.currentTime - state.startedAtAudioTime) * 1000;
  let currentIndex = -1;
  let spokenCount = 0;

  for (let i = 0; i < state.timeline.length; i += 1) {
    const item = state.timeline[i];
    if (elapsedMs >= item.endMs) {
      spokenCount = i + 1;
    } else if (elapsedMs >= item.startMs) {
      currentIndex = i;
      break;
    }
  }

  if (currentIndex < 0 && spokenCount < state.timeline.length) currentIndex = spokenCount;

  state.wordSpans.forEach((span, index) => {
    span.classList.toggle("spoken", index < spokenCount);
    span.classList.toggle("current", index === currentIndex);
  });
}

function finishIfIdle() {
  if (!state.done || state.currentSource || state.queue.length) return;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  sendEl.disabled = false;
  setStatus("Idle");
  updateHighlight();
}

function playNext() {
  if (state.currentSource || !state.queue.length) {
    finishIfIdle();
    return;
  }

  const chunk = state.queue.shift();
  const chunkStartMs = state.totalAudioMs;
  state.totalAudioMs += chunk.durationMs;
  for (const item of chunk.alignments) {
    state.timeline.push({
      word: item.word,
      startMs: chunkStartMs + item.startMs,
      endMs: chunkStartMs + item.endMs,
    });
  }

  const source = state.audioContext.createBufferSource();
  source.buffer = chunk.buffer;
  source.connect(state.audioContext.destination);
  source.onended = () => {
    state.currentSource = null;
    playNext();
  };
  state.currentSource = source;
  if (!state.playing) {
    state.startedAtAudioTime = state.audioContext.currentTime;
    state.playing = true;
  }
  source.start();

  if (!state.timer) state.timer = setInterval(updateHighlight, 35);
}

async function enqueueAudio(payload) {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  if (state.audioContext.state === "suspended") await state.audioContext.resume();
  const buffer = pcmToAudioBuffer(payload.audioBase64, payload.sampleRate);
  if (!buffer) return;
  state.queue.push({
    buffer,
    durationMs: payload.durationMs || (buffer.duration * 1000),
    alignments: payload.alignments || [],
  });
  playNext();
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
  for (const voice of model.voices) {
    const option = document.createElement("option");
    option.value = voice;
    option.textContent = voice;
    voiceEl.appendChild(option);
  }
  if (model.voices.includes("magda_001")) voiceEl.value = "magda_001";
}

function syncProtocolDefaults() {
  if (protocolEl.value === "grpc" && portEl.value === "8000") portEl.value = "50051";
  if (protocolEl.value === "websocket" && portEl.value === "50051") portEl.value = "8000";
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
  buildHighlight(text);
  firstByteEl.textContent = "-";
  sendEl.disabled = true;
  setStatus("Starting");
  await window.tinfer.startSynthesis({
    protocol: protocolEl.value,
    host: hostEl.value,
    port: portEl.value,
    modelId: modelEl.value,
    voiceId: voiceEl.value,
    text,
  });
}

async function stop() {
  resetPlayback();
  await window.tinfer.stopSynthesis();
  sendEl.disabled = false;
  setStatus("Stopped");
}

modelEl.addEventListener("change", populateVoices);
protocolEl.addEventListener("change", syncProtocolDefaults);
sendEl.addEventListener("click", send);
stopEl.addEventListener("click", stop);

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
    resetPlayback();
    sendEl.disabled = false;
    setStatus("Stopped");
    return;
  }
  if (event.type === "error") {
    resetPlayback();
    sendEl.disabled = false;
    setStatus(event.message || "Error", true);
  }
});

window.tinfer.getCatalog().then(populateCatalog).catch(() => {
  populateCatalog([{ id: "magda", voices: ["magda_001"] }]);
});
