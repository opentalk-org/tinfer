let activeRun = null;

function createRun() {
  if (activeRun) activeRun.cancel();
  const run = {
    cancelled: false,
    firstByteMs: null,
    sendChunk() {},
    force() {},
    end() {},
    cancel() { run.cancelled = true; },
  };
  activeRun = run;
  return run;
}

function currentRun() {
  return activeRun;
}

function clearRun(run) {
  if (activeRun === run) activeRun = null;
}

function audioEvent(run, startedAt, audioBase64, alignments, sampleRate = 24000) {
  const firstByteMs = run.firstByteMs === null ? Math.round(performance.now() - startedAt) : null;
  if (firstByteMs !== null) run.firstByteMs = firstByteMs;
  return {
    type: "audio",
    audioBase64,
    sampleRate,
    durationMs: (Buffer.from(audioBase64, "base64").length / 2 / sampleRate) * 1000,
    alignments,
    firstByteMs,
  };
}

module.exports = { audioEvent, clearRun, createRun, currentRun };
