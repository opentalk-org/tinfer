function setStatus(value, isError = false) {
  statusEl.textContent = value;
  statusEl.classList.toggle("error", isError);
}

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

