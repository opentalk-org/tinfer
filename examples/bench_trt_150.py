import sys, time, statistics, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tinfer.core.engine import StreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig

BASE = Path("/workspace/converted_models")
MODEL_ID = "styletts2"
MODEL_NAME = "libri"
VOICE_ID = "libri_f1"
MODEL_PATH = BASE / MODEL_NAME / "model.pth"
VOICES = str(BASE / MODEL_NAME / "voices")

TEXT = ("The quick brown fox jumps over the lazy dog while the morning sun rose "
        "gently above the calm and quiet river beside the old grey stone bridge in town.")
assert len(TEXT) == 150, len(TEXT)

def bench(engine_name):
    cfg = StreamingTTSConfig(compile_models=False, runtime_engine=engine_name,
                             default_batch_size=1)
    tts = StreamingTTS(cfg)
    print(f"\n### runtime_engine = {engine_name}")
    print(f"Loading {MODEL_PATH}")
    tts.load_model(MODEL_ID, str(MODEL_PATH), voices_folder=VOICES)
    tts.warmup([MODEL_ID], [VOICE_ID], num_warmup_tasks=1)

    async def gen():
        return await tts.generate_full(MODEL_ID, VOICE_ID, TEXT, {})

    # warmup a couple of real gens
    for _ in range(3):
        asyncio.run(gen())

    lat = []
    samples = 0
    sr = 24000
    N = 20
    for _ in range(N):
        t0 = time.perf_counter()
        chunk = asyncio.run(gen())
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)
        samples = len(chunk.audio)
        sr = chunk.sample_rate

    audio_s = samples / sr
    lat.sort()
    mean = statistics.mean(lat)
    p50 = statistics.median(lat)
    p90 = lat[int(0.9 * (N - 1))]
    print(f"audio produced : {samples} samples = {audio_s:.2f} s @ {sr} Hz")
    print(f"latency mean   : {mean:.1f} ms")
    print(f"latency p50    : {p50:.1f} ms")
    print(f"latency p90    : {p90:.1f} ms")
    print(f"latency min/max: {min(lat):.1f} / {max(lat):.1f} ms")
    print(f"RTF            : {(mean/1000.0)/audio_s:.4f}  ({audio_s/(mean/1000.0):.0f}x real-time)")
    tts.stop()
    return mean, audio_s

if __name__ == "__main__":
    engines = sys.argv[1:] or ["tensorrt"]
    for e in engines:
        bench(e)
