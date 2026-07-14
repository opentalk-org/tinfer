import asyncio
from pathlib import Path

from tinfer import AsyncStreamingTTS, StreamingTTS, StreamingTTSConfig


ARTIFACT = Path(__file__).parents[2] / "tests/artifacts/stub"
MODEL = {"id": "stub", "path": str(ARTIFACT), "backend": "onnx", "device": "cpu", "max_batch": 4}


def test_dynamic_loading_and_stream_methods_use_rust():
    tts = StreamingTTS(StreamingTTSConfig(models=[MODEL]))
    stream = tts.create_stream("stub", "default", {})
    assert stream.get_audio() == []
    assert stream.get_state() == {}
    stream.add_text("hello")
    stream.try_generate()
    stream.force_generate()
    assert stream.collect_audio()[0].sample_rate == 24_000
    assert stream.get_state() == {"text": "hello"}
    stream.close()

    tts.unload_model("stub")
    tts.load_model(MODEL)
    assert tts.get_model_ids() == ["stub"]
    assert asyncio.run(tts.generate_full("stub", "default", "full", {})).audio.size > 0
    asyncio.run(tts.async_warmup(["stub"], ["default"], 2))
    tts.run()
    tts.stop()


def test_async_facade_delegates_the_real_interface():
    async def exercise():
        tts = AsyncStreamingTTS(StreamingTTSConfig(models=[MODEL]))
        assert tts.get_voice_ids("stub") == ["default"]
        chunks = [chunk async for chunk in tts.generate("stub", "default", "async", {})]
        assert chunks[0].audio.size > 0
        tts.unload_model("stub")
        tts.load_model(MODEL)
        await tts.async_warmup(["stub"], ["default"], 1)
        tts.stop()

    asyncio.run(exercise())
