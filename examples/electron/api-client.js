const WebSocket = require("ws");
const { StringDecoder } = require("node:string_decoder");

const { fromHttp } = require("./catalog");

const SAMPLE_RATE = 24000;
const OUTPUT_FORMAT = "pcm_24000";
const API_MODES = new Set(["ws_single", "ws_multi", "post_audio", "post_timing", "stream_audio", "stream_timing"]);

function endpointPath(mode, voiceId) {
  const base = `/v1/text-to-speech/${encodeURIComponent(voiceId)}`;
  const suffixes = {
    ws_single: "/stream-input",
    ws_multi: "/multi-stream-input",
    post_audio: "",
    post_timing: "/with-timestamps",
    stream_audio: "/stream",
    stream_timing: "/stream/with-timestamps",
  };
  if (!API_MODES.has(mode)) throw new Error(`unknown API mode: ${mode}`);
  return `${base}${suffixes[mode]}`;
}

function makeSpeechBody(request) {
  return {
    text: request.text,
    model_id: request.modelId,
    language_code: request.language,
    voice_settings: { alpha: request.alpha, beta: request.beta, speed: request.speed },
  };
}

function initializationMessage(request) {
  return {
    text: " ",
    voice_settings: { alpha: request.alpha, beta: request.beta, speed: request.speed },
    generation_config: { chunk_length_schedule: request.chunkSchedule },
  };
}

function speechTextMessage(request) {
  return { text: request.text.endsWith(" ") ? request.text : `${request.text} ` };
}

function makeSingleMessages(request) {
  return {
    open: [initializationMessage(request), speechTextMessage(request)],
    flush: { text: " ", flush: true },
    end: { text: "" },
  };
}

function makeMultiMessages(contextId, request) {
  const initial = initializationMessage(request);
  const text = speechTextMessage(request);
  return {
    open: [{ ...initial, context_id: contextId }, { ...text, context_id: contextId }],
    flush: { text: " ", context_id: contextId, flush: true },
    end: { context_id: contextId, close_context: true },
    close: { close_socket: true },
  };
}

function groupCharacters(items, granularity) {
  if (granularity === "char") return items.filter((item) => !/\s/.test(item.word));
  const words = [];
  for (const item of items) {
    if (/\s/.test(item.word)) continue;
    const previous = words[words.length - 1];
    if (previous && previous.endMs === item.startMs) {
      previous.word += item.word;
      previous.endMs = item.endMs;
    } else {
      words.push({ ...item });
    }
  }
  return words;
}

function timingAlignments(alignment, granularity = "char") {
  if (!alignment) return [];
  const chars = alignment.characters || alignment.chars;
  const starts = alignment.character_start_times_seconds || alignment.char_start_times_seconds;
  const ends = alignment.character_end_times_seconds || alignment.char_end_times_seconds;
  const items = chars.map((word, index) => ({ word, startMs: starts[index] * 1000, endMs: ends[index] * 1000 }));
  return groupCharacters(items, granularity);
}

class PcmReader {
  constructor() { this.pending = Buffer.alloc(0); }
  push(bytes) {
    this.pending = Buffer.concat([this.pending, Buffer.from(bytes)]);
    const length = this.pending.length - (this.pending.length % 2);
    if (!length) return [];
    const chunk = this.pending.subarray(0, length);
    this.pending = this.pending.subarray(length);
    return [chunk];
  }
  finish() {
    if (this.pending.length) throw new Error("PCM stream ended inside a sample boundary");
  }
}

class TimingReader {
  constructor(granularity = "char") {
    this.pending = "";
    this.granularity = granularity;
    this.decoder = new StringDecoder("utf8");
  }
  push(bytes) {
    this.pending += this.decoder.write(Buffer.from(bytes));
    const lines = this.pending.split("\n");
    this.pending = lines.pop();
    return lines.filter(Boolean).map((line) => {
      const row = JSON.parse(line);
      return { audioBase64: row.audio_base64, alignments: timingAlignments(row.normalized_alignment || row.alignment, this.granularity) };
    });
  }
  finish() {
    this.pending += this.decoder.end();
    if (this.pending) throw new Error("timing stream ended inside a JSON record");
  }
}

function normalizeHostPort(host, port) {
  const cleanHost = String(host).replace(/^https?:\/\//, "").replace(/^wss?:\/\//, "").replace(/\/.*$/, "").trim();
  if (!cleanHost) throw new Error("host is required");
  return cleanHost.includes(":") ? cleanHost : `${cleanHost}:${port}`;
}

function queryString(mode, request) {
  if (mode === "ws_single" || mode === "ws_multi") {
    return new URLSearchParams({
      model_id: request.modelId,
      output_format: OUTPUT_FORMAT,
      language_code: request.language,
      sync_alignment: "true",
    }).toString();
  }
  if (!API_MODES.has(mode)) throw new Error(`unknown API mode: ${mode}`);
  return new URLSearchParams({ output_format: OUTPUT_FORMAT }).toString();
}

function durationMs(base64) {
  return (Buffer.from(base64, "base64").length / 2 / SAMPLE_RATE) * 1000;
}

async function fetchCatalog(host, port) {
  const base = `http://${normalizeHostPort(host, port)}`;
  const [modelsResponse, voicesResponse] = await Promise.all([fetch(`${base}/v1/models`), fetch(`${base}/v1/voices`)]);
  if (!modelsResponse.ok || !voicesResponse.ok) {
    throw new Error(`catalog failed: models ${modelsResponse.status}, voices ${voicesResponse.status}`);
  }
  return fromHttp(await modelsResponse.json(), await voicesResponse.json());
}

function socketAlignments(alignment, granularity) {
  if (!alignment) return [];
  const chars = alignment.chars || alignment.characters;
  const starts = alignment.charStartTimesMs || alignment.characterStartTimesMs;
  const durations = alignment.charDurationsMs || alignment.characterDurationsMs;
  const items = chars.map((word, index) => ({ word, startMs: Number(starts[index]), endMs: Number(starts[index]) + Number(durations[index]) }));
  return groupCharacters(items, granularity);
}

function socketStart(request, run, emit, finish) {
  const contextId = `run-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const messages = request.mode === "ws_multi" ? makeMultiMessages(contextId, request) : makeSingleMessages(request);
  const target = normalizeHostPort(request.host, request.port);
  const url = `ws://${target}${endpointPath(request.mode, request.voiceId)}?${queryString(request.mode, request)}`;
  const socket = new WebSocket(url, { maxPayload: 10 * 1024 * 1024 });
  let contextClosed = false;
  run.sendChunk = (text) => {
    const spacedText = text.endsWith(" ") ? text : `${text} `;
    const payload = request.mode === "ws_multi" ? { text: spacedText, context_id: contextId } : { text: spacedText };
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  };
  run.force = () => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify(messages.flush));
  run.end = () => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify(messages.end));
  run.cancel = () => { run.cancelled = true; socket.close(); };
  socket.on("open", () => messages.open.forEach((message) => socket.send(JSON.stringify(message))));
  socket.on("message", (bytes) => {
    try {
      const message = JSON.parse(String(bytes));
      if (message.error) throw new Error(message.error);
      if (message.audio) emit(message.audio, socketAlignments(message.normalizedAlignment || message.alignment, request.granularity));
      if (!message.isFinal) return;
      if (request.mode === "ws_multi" && !contextClosed) {
        contextClosed = true;
        socket.send(JSON.stringify(messages.close));
        return;
      }
      socket.close();
    } catch (error) {
      finish(error);
      socket.close();
    }
  });
  socket.on("error", finish);
  socket.on("close", () => finish());
}

function audioFromTiming(row, granularity) {
  return {
    audioBase64: row.audio_base64,
    alignments: timingAlignments(row.normalized_alignment || row.alignment, granularity),
  };
}

async function consumeStream(response, reader, emit) {
  const stream = response.body.getReader();
  for (;;) {
    const result = await stream.read();
    if (result.done) break;
    for (const item of reader.push(result.value)) {
      if (Buffer.isBuffer(item)) emit(item.toString("base64"), []);
      else emit(item.audioBase64, item.alignments);
    }
  }
  reader.finish();
}

async function httpStart(request, run, emit, finish) {
  const target = normalizeHostPort(request.host, request.port);
  const url = `http://${target}${endpointPath(request.mode, request.voiceId)}?${queryString(request.mode, request)}`;
  const controller = new AbortController();
  run.cancel = () => { run.cancelled = true; controller.abort(); };
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(makeSpeechBody(request)),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`synthesis failed: ${response.status} ${await response.text()}`);
    if (request.mode === "post_audio") emit(Buffer.from(await response.arrayBuffer()).toString("base64"), []);
    if (request.mode === "post_timing") {
      const item = audioFromTiming(await response.json(), request.granularity);
      emit(item.audioBase64, item.alignments);
    }
    if (request.mode === "stream_audio") await consumeStream(response, new PcmReader(), emit);
    if (request.mode === "stream_timing") await consumeStream(response, new TimingReader(request.granularity), emit);
    finish();
  } catch (error) {
    finish(error);
  }
}

function start(request, run, emit, finish) {
  if (request.mode === "ws_single" || request.mode === "ws_multi") socketStart(request, run, emit, finish);
  else httpStart(request, run, emit, finish);
}

module.exports = {
  API_MODES, SAMPLE_RATE, OUTPUT_FORMAT, WebSocket, PcmReader, TimingReader,
  endpointPath, makeSpeechBody, makeSingleMessages, makeMultiMessages,
  timingAlignments, normalizeHostPort, queryString, durationMs, fetchCatalog, start,
};
