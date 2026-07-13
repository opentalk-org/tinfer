import asyncio
import unittest
from unittest.mock import AsyncMock

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from tinfer.server.health import HealthState
from tinfer.server.websocket.schemas import SpeechOutputFormat
from tinfer.server.websocket.server import WebSocketServer
from tinfer.server.websocket.stream_context import StreamContext
from tmp_tests.tts_api.fakes import DelayedCleanupTTS, FakeTTS, PumpFailureStream


class WebSocketSingleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tts = FakeTTS()
        self.health = HealthState(warmup_complete=True)
        app = WebSocketServer(self.tts, health=self.health).create_app()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_streams_audio_flushes_and_finishes_once(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input"
            "?model_id=libri&language_code=en-us&output_format=pcm_24000&sync_alignment=true"
        )
        await ws.send_json(
            {
                "text": " ",
                "voice_settings": {"speed": 1.1},
                "generation_config": {"chunk_length_schedule": [80]},
            }
        )
        await ws.send_json({"text": "Hello ", "flush": True})
        first = await asyncio.wait_for(ws.receive_json(), timeout=1)
        second = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertFalse(first["isFinal"])
        self.assertFalse(second["isFinal"])
        self.assertIn("charStartTimesMs", first["alignment"])
        self.assertEqual(self.tts.streams[0].force_count, 1)

        await ws.send_json({"text": ""})
        final = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertEqual(final, {"isFinal": True})
        self.assertEqual((await ws.receive()).type, WSMsgType.CLOSE)
        for _ in range(10):
            if self.health.active_connections == 0:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.tts.streams[0].close_count, 1)
        self.assertEqual(self.health.active_connections, 0)

    async def test_rejects_invalid_initial_and_text_messages(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": "Hello "})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("first message", error["error"])
        await ws.close()

        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": " "})
        await ws.send_json({"text": "missing trailing space"})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("end in a space", error["error"])

    async def test_trigger_is_conditional_and_client_final_flag_is_rejected(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": " ", "generation_config": {"chunk_length_schedule": [80]}})
        await ws.send_json({"text": "Hello ", "try_trigger_generation": True})
        await asyncio.wait_for(ws.receive_json(), timeout=1)
        await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertEqual(self.tts.streams[0].force_count, 0)
        self.assertEqual(self.tts.streams[0].try_count, 1)
        await ws.send_json({"isFinal": True})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("unsupported message field", error["error"])

        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": " "})
        await ws.send_json({"voice_settings": {"speed": 1.0}})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("text is required", error["error"])

    async def test_accepts_unchanged_settings_and_rejects_changed_settings(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        settings = {"speed": 1.1}
        generator = {"chunk_length_schedule": [80]}
        await ws.send_json(
            {
                "text": " ",
                "voice_settings": settings,
                "generation_config": generator,
                "xi-api-key": "ignored",
                "authorization": "ignored",
                "pronunciation_dictionary_locators": [],
            }
        )
        await ws.send_json(
            {
                "text": "Hello ",
                "flush": True,
                "voice_settings": settings,
                "generator_config": generator,
            }
        )
        await asyncio.wait_for(ws.receive_json(), timeout=1)
        await asyncio.wait_for(ws.receive_json(), timeout=1)
        await ws.send_json({"text": "Again ", "voice_settings": {"speed": 1.2}})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("cannot change", error["error"])

    async def test_rejects_unknown_message_fields(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": " "})
        await ws.send_json({"text": "Hello ", "unknown": True})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("unsupported message field", error["error"])

    async def test_inactivity_closes_stream(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/stream-input"
            "?model_id=libri&output_format=pcm_24000&inactivity_timeout=1"
        )
        await ws.send_json({"text": " "})
        error = await asyncio.wait_for(ws.receive_json(), timeout=2)
        self.assertIn("inactivity", error["error"])
        await asyncio.wait_for(ws.receive(), timeout=1)
        self.assertEqual(self.tts.streams[0].close_count, 1)

    async def test_audio_pump_failure_unblocks_finalization(self) -> None:
        stream = PumpFailureStream([])
        context = StreamContext(
            stream=stream,
            output_format=SpeechOutputFormat.PCM_24000,
            send_response=AsyncMock(),
            inactivity_timeout=20,
            close_response=AsyncMock(),
        )
        await context.add_text("Failure case ")

        with self.assertRaisesRegex(RuntimeError, "audio pump failed"):
            await asyncio.wait_for(context.finalize(), timeout=0.2)
        self.assertEqual(stream.close_count, 1)

    async def test_disconnect_cleanup_completes_before_health_release(self) -> None:
        tts = DelayedCleanupTTS()
        health = HealthState(warmup_complete=True)
        client = TestClient(TestServer(WebSocketServer(tts, health=health).create_app()))
        await client.start_server()
        ws = await client.ws_connect(
            "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": " "})
        await ws.send_json({"text": "Start pump ", "flush": True})
        for _ in range(20):
            if tts.streams:
                break
            await asyncio.sleep(0.01)
        stream = tts.streams[0]
        await asyncio.wait_for(stream.pull_started.wait(), timeout=1)

        await ws.close()
        await asyncio.wait_for(stream.cleanup_started.wait(), timeout=1)
        self.assertEqual(health.active_connections, 1)
        stream.cleanup_release.set()
        for _ in range(20):
            if health.active_connections == 0:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(health.active_connections, 0)
        await client.close()


if __name__ == "__main__":
    unittest.main()
