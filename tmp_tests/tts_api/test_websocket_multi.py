import asyncio
import unittest

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from tinfer.server.health import HealthState
from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeTTS


class WebSocketMultiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tts = FakeTTS()
        self.health = HealthState(warmup_complete=True)
        app = WebSocketServer(self.tts, health=self.health).create_app()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_interleaves_isolated_contexts_and_closes_socket_after_finals(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input"
            "?model_id=libri&language_code=en-us&output_format=pcm_24000"
        )
        await ws.send_json({"context_id": "a", "text": " "})
        await ws.send_json({"context_id": "a", "text": "Alpha ", "flush": True})
        await ws.send_json({"context_id": "b", "text": "Beta ", "flush": True})

        audio = [await asyncio.wait_for(ws.receive_json(), timeout=1) for _ in range(4)]
        self.assertEqual([item["contextId"] for item in audio], ["a", "a", "b", "b"])
        self.assertTrue(all(not item["isFinal"] for item in audio))
        self.assertEqual([stream.texts for stream in self.tts.streams], [["Alpha "], ["Beta "]])

        await ws.send_json({"context_id": "a", "text": "", "close_context": True})
        final_a = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertEqual(final_a, {"contextId": "a", "isFinal": True})
        await ws.send_json({"context_id": "a", "text": "Again ", "flush": True})
        reused = [await asyncio.wait_for(ws.receive_json(), timeout=1) for _ in range(2)]
        self.assertEqual({item["contextId"] for item in reused}, {"a"})

        await ws.send_json({"close_socket": True})
        finals = [await asyncio.wait_for(ws.receive_json(), timeout=1) for _ in range(2)]
        self.assertEqual({item["contextId"] for item in finals}, {"a", "b"})
        self.assertTrue(all(item["isFinal"] for item in finals))
        self.assertEqual((await ws.receive()).type, WSMsgType.CLOSE)
        self.assertTrue(all(stream.close_count == 1 for stream in self.tts.streams))

    async def test_requires_initialization_and_rejects_invalid_transition(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"context_id": "a", "text": "Not initialization "})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("first message", error["error"])

        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"context_id": "a", "text": " "})
        await ws.send_json({"context_id": "missing", "close_context": True})
        error = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertIn("unknown context", error["error"])

    async def test_keepalive_does_not_finalize_context(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input"
            "?model_id=libri&output_format=pcm_24000&inactivity_timeout=1"
        )
        await ws.send_json({"context_id": "a", "text": " "})
        await asyncio.sleep(0.03)
        await ws.send_json({"context_id": "a", "text": ""})
        await asyncio.sleep(0.03)
        self.assertFalse(ws.closed)
        await ws.send_json({"close_socket": True})
        self.assertEqual((await ws.receive_json())["contextId"], "a")

    async def test_default_context_flush_without_text_and_alignment_setting(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"text": " "})
        await ws.send_json({"flush": True})
        first = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertEqual(first["contextId"], "default")
        self.assertEqual(self.tts.params[0]["alignment_type"].value, "none")
        await ws.send_json({"close_socket": True})

    async def test_context_generation_config_sets_chunk_schedule(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json(
            {
                "context_id": "scheduled",
                "text": " ",
                "generation_config": {"chunk_length_schedule": [80, 140]},
            }
        )
        for _ in range(20):
            if self.tts.params:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.tts.params[0]["chunk_length_schedule"], [80, 140])
        await ws.send_json({"close_socket": True})
        await ws.receive_json()

    async def test_live_settings_reinitialize_context(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"context_id": "one", "text": " "})
        await ws.send_json(
            {
                "context_id": "one",
                "text": " ",
                "voice_settings": {"speed": 1.1},
            }
        )
        final = await asyncio.wait_for(ws.receive_json(), timeout=1)
        while not final["isFinal"]:
            final = await asyncio.wait_for(ws.receive_json(), timeout=1)
        self.assertTrue(final["isFinal"])
        self.assertEqual(final["contextId"], "one")
        self.assertEqual(len(self.tts.streams), 2)
        self.assertEqual(self.tts.streams[0].close_count, 1)

    async def test_disconnect_closes_each_context_once(self) -> None:
        ws = await self.client.ws_connect(
            "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
        )
        await ws.send_json({"context_id": "a", "text": " "})
        for _ in range(20):
            if self.tts.streams:
                break
            await asyncio.sleep(0.01)
        await ws.close()
        for _ in range(20):
            if self.tts.streams[0].close_count == 1:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.tts.streams[0].close_count, 1)


if __name__ == "__main__":
    unittest.main()
