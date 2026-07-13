const assert = require("node:assert/strict");
const test = require("node:test");

const {
  endpointPath,
  makeSpeechBody,
  makeSingleMessages,
  makeMultiMessages,
  queryString,
  PcmReader,
  TimingReader,
} = require("../api-client");

test("maps six modes to exact endpoint paths", () => {
  const voice = "voice / one";
  assert.equal(endpointPath("ws_single", voice), "/v1/text-to-speech/voice%20%2F%20one/stream-input");
  assert.equal(endpointPath("ws_multi", voice), "/v1/text-to-speech/voice%20%2F%20one/multi-stream-input");
  assert.equal(endpointPath("post_audio", voice), "/v1/text-to-speech/voice%20%2F%20one");
  assert.equal(endpointPath("post_timing", voice), "/v1/text-to-speech/voice%20%2F%20one/with-timestamps");
  assert.equal(endpointPath("stream_audio", voice), "/v1/text-to-speech/voice%20%2F%20one/stream");
  assert.equal(endpointPath("stream_timing", voice), "/v1/text-to-speech/voice%20%2F%20one/stream/with-timestamps");
});

test("uses exact query keys for all six modes", () => {
  const request = { modelId: "libri", language: "en-us" };
  const expectedSocket = {
    model_id: "libri", output_format: "pcm_24000", language_code: "en-us", sync_alignment: "true",
  };
  for (const mode of ["ws_single", "ws_multi"]) {
    assert.deepEqual(Object.fromEntries(new URLSearchParams(queryString(mode, request))), expectedSocket);
  }
  for (const mode of ["post_audio", "post_timing", "stream_audio", "stream_timing"]) {
    assert.deepEqual(Object.fromEntries(new URLSearchParams(queryString(mode, request))), { output_format: "pcm_24000" });
  }
});

test("builds request body from selected model, language, and controls", () => {
  const body = makeSpeechBody({ text: "Hello", modelId: "libri", language: "en-us", alpha: 0.3, beta: 0.7, speed: 1.1, chunkSchedule: [20, 40] });
  assert.deepEqual(body, {
    text: "Hello",
    model_id: "libri",
    language_code: "en-us",
    voice_settings: { alpha: 0.3, beta: 0.7, speed: 1.1 },
  });
});

test("creates single-context initialization, flush, and end messages", () => {
  const messages = makeSingleMessages({ text: "Hello ", modelId: "libri", language: "en-us", alpha: 0.3, beta: 0.7, speed: 1, chunkSchedule: [20] });
  assert.deepEqual(messages.open, [{
    text: " ",
    voice_settings: { alpha: 0.3, beta: 0.7, speed: 1 },
    generation_config: { chunk_length_schedule: [20] },
  }, { text: "Hello " }]);
  assert.deepEqual(messages.flush, { text: " ", flush: true });
  assert.deepEqual(messages.end, { text: "" });
});

test("creates context-aware initialization and graceful close messages", () => {
  const messages = makeMultiMessages("run-4", { text: "Hello ", modelId: "libri", language: "en-us", alpha: 0.3, beta: 0.7, speed: 1, chunkSchedule: [20] });
  assert.deepEqual(messages.open, [{
    text: " ", context_id: "run-4",
    voice_settings: { alpha: 0.3, beta: 0.7, speed: 1 },
    generation_config: { chunk_length_schedule: [20] },
  }, { text: "Hello ", context_id: "run-4" }]);
  assert.deepEqual(messages.flush, { text: " ", context_id: "run-4", flush: true });
  assert.deepEqual(messages.end, { context_id: "run-4", close_context: true });
  assert.deepEqual(messages.close, { close_socket: true });
});

test("preserves raw PCM split boundaries", () => {
  const reader = new PcmReader();
  assert.equal(reader.push(Buffer.from([1])).length, 0);
  assert.deepEqual(reader.push(Buffer.from([2, 3])), [Buffer.from([1, 2])]);
  assert.throws(() => reader.finish(), /sample boundary/);
});

test("preserves newline JSON split boundaries and converts seconds to milliseconds", () => {
  const reader = new TimingReader();
  assert.deepEqual(reader.push(Buffer.from('{"audio_base64":"AQI=","alignment":{"characters":["H"],')), []);
  const rows = reader.push(Buffer.from('"character_start_times_seconds":[0.1],"character_end_times_seconds":[0.2]}}\n'));
  assert.deepEqual(rows[0].alignments, [{ word: "H", startMs: 100, endMs: 200 }]);
  assert.equal(rows[0].audioBase64, "AQI=");
  reader.finish();
});

test("groups character timing into words for word highlighting", () => {
  const reader = new TimingReader("word");
  const row = {
    audio_base64: "AQI=",
    alignment: {
      characters: ["H", "i", " ", "a"],
      character_start_times_seconds: [0, 0.1, 0.2, 0.3],
      character_end_times_seconds: [0.1, 0.2, 0.3, 0.4],
    },
  };
  const [result] = reader.push(Buffer.from(`${JSON.stringify(row)}\n`));
  assert.deepEqual(result.alignments, [
    { word: "Hi", startMs: 0, endMs: 200 },
    { word: "a", startMs: 300, endMs: 400 },
  ]);
});

test("decodes newline JSON split inside a multibyte alignment character", () => {
  const reader = new TimingReader("char");
  const row = Buffer.from(`${JSON.stringify({
    audio_base64: "AQI=",
    alignment: {
      characters: ["ą"],
      character_start_times_seconds: [0],
      character_end_times_seconds: [0.1],
    },
  })}\n`);
  const marker = Buffer.from("ą");
  const split = row.indexOf(marker) + 1;

  assert.deepEqual(reader.push(row.subarray(0, split)), []);
  const [result] = reader.push(row.subarray(split));
  assert.deepEqual(result.alignments, [{ word: "ą", startMs: 0, endMs: 100 }]);
  reader.finish();
});
