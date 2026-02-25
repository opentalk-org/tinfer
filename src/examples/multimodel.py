import asyncio
import numpy as np
import soundfile as sf

from tinfer.core.engine import StreamingTTS
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig

from config import base_dir

MODELS = [
    ("agnieszka", "66a4ecf82e2a7ae68b14add9_7.97_4.27"),
    ("magda", "magda_001"),
    ("olam", "any"),
]

SAMPLE_TEXTS = [
    "To jest pierwszy przykład syntezy mowy z wieloma modelami.",
    "Drugi fragment demonstruje przeplatanie żądań między modelami.",
    "Trzeci tekst kończy zestaw próbek dla każdego głosu.",
]

OUTPUT_DIR = base_dir / "output_wavs" / "multimodel"


def save_audio_chunks(chunks, output_path, sample_rate):
    audio_arrays = [chunk.audio for chunk in chunks]
    concatenated = np.concatenate(audio_arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), concatenated, sample_rate)


def load_models(tts):
    for name, voice_id in MODELS:
        model_path = base_dir / "converted_models" / name / "model.pth"
        voices_folder = str(base_dir / "converted_models" / name / "voices")
        print(f"Loading model {name} from {model_path}")
        tts.load_model(name, str(model_path), voices_folder=voices_folder)
    model_ids = [m[0] for m in MODELS]
    voice_ids = [m[1] for m in MODELS]
    tts.warmup(model_ids, voice_ids)


async def generate_one(async_tts, model_id, voice_id, text, params, out_path):
    chunks = []
    async for chunk in async_tts.generate(model_id, voice_id, text, params):
        chunks.append(chunk)
    if chunks:
        save_audio_chunks(chunks, out_path, chunks[0].sample_rate)
    return out_path


async def run_intertwined(async_tts):
    tasks = []
    for sample_idx in range(3):
        for model_idx, (model_id, voice_id) in enumerate(MODELS):
            text = SAMPLE_TEXTS[sample_idx]
            out_path = OUTPUT_DIR / f"{model_id}_sample_{sample_idx + 1:03d}.wav"
            tasks.append(
                generate_one(async_tts, model_id, voice_id, text, {}, out_path)
            )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            raise r
    return results


async def main():
    config = StreamingTTSConfig()
    tts = StreamingTTS(config)
    load_models(tts)
    async_tts = AsyncStreamingTTS(tts)
    print("Running 9 intertwined requests (3 per model)...")
    await run_intertwined(async_tts)
    async_tts.stop()
    tts.stop()
    
    print(f"Saved 3 samples per model to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
