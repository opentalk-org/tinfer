const path = require("node:path");
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");

const { fromGrpc } = require("./catalog");

const definition = protoLoader.loadSync(path.join(__dirname, "styletts.proto"), {
  keepCase: true, longs: String, enums: String, defaults: true, oneofs: true,
});
const service = grpc.loadPackageDefinition(definition).styletts.v1;
const options = {
  "grpc.max_receive_message_length": 64 * 1024 * 1024,
  "grpc.max_send_message_length": 64 * 1024 * 1024,
};

function invoke(client, method, request) {
  return new Promise((resolve, reject) => {
    client[method](request, (error, response) => (error ? reject(error) : resolve(response)));
  });
}

function config(request) {
  return { model_id: request.modelId, voice_id: request.voiceId, sample_rate_hz: 24000, language: request.language };
}

function alignments(response) {
  return response.alignments.map((item) => ({
    word: item.word,
    startMs: Number(item.start_ms),
    endMs: Number(item.end_ms),
  }));
}

function createClient(address) {
  return new service.StyleTTSService(address, grpc.credentials.createInsecure(), options);
}

async function fetchCatalog(address) {
  const client = createClient(address);
  const [models, voices] = await Promise.all([
    invoke(client, "ListModels", {}),
    invoke(client, "ListVoices", {}),
  ]);
  client.close();
  return fromGrpc(models, voices);
}

function bindReadSide(call, run, emit, finish) {
  call.on("data", (response) => emit(Buffer.from(response.audio_data).toString("base64"), alignments(response)));
  call.on("error", (error) => finish(error));
  call.on("end", () => finish());
}

function start(request, run, emit, finish) {
  const client = createClient(request.address);
  const payload = { text: request.text, config: config(request) };
  if (request.mode === "unary") {
    const call = client.Synthesize(payload, (error, response) => {
      if (!error) emit(Buffer.from(response.audio_data).toString("base64"), alignments(response));
      client.close();
      finish(error);
    });
    run.cancel = () => { run.cancelled = true; call.cancel(); client.close(); };
    return;
  }
  if (request.mode === "stream") {
    const call = client.SynthesizeStream(payload);
    run.cancel = () => { run.cancelled = true; call.cancel(); client.close(); };
    bindReadSide(call, run, emit, (error) => { client.close(); finish(error); });
    return;
  }
  const call = client.SynthesizeIncremental();
  call.write({ config: config(request) });
  call.write({ text_chunk: request.text });
  run.sendChunk = (text) => call.write({ text_chunk: text });
  run.force = () => call.write({ force_synthesis: {} });
  run.end = () => call.end();
  run.cancel = () => { run.cancelled = true; call.cancel(); client.close(); };
  bindReadSide(call, run, emit, (error) => { client.close(); finish(error); });
}

module.exports = { fetchCatalog, start };
