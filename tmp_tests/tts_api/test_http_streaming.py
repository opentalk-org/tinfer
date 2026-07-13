import asyncio
import json
import unittest

from aiohttp.test_utils import TestClient, TestServer

from tinfer.server.health import HealthState
from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeTTS


class HttpStreamingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.release = asyncio.Event()
        self.tts = FakeTTS(self.release)
        self.health = HealthState(warmup_complete=True)
        app = WebSocketServer(self.tts, health=self.health).create_app()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        self.release.set()
        await self.client.close()

    async def test_plain_stream_delivers_first_chunk_before_completion(self) -> None:
        response = await self.client.post(
            "/v1/text-to-speech/voice/stream?output_format=pcm_24000",
            json={"text": "AB", "model_id": "libri"},
        )
        first = await asyncio.wait_for(response.content.readexactly(4), timeout=1)
        self.assertEqual(len(first), 4)
        self.assertEqual(self.health.active_connections, 1)
        self.release.set()
        self.assertEqual(len(await response.read()), 4)
        await asyncio.sleep(0)
        self.assertEqual(self.tts.streams[0].close_count, 1)
        self.assertEqual(self.health.active_connections, 0)

    async def test_timing_stream_is_ordered_newline_json(self) -> None:
        response = await self.client.post(
            "/v1/text-to-speech/voice/stream/with-timestamps?output_format=pcm_24000",
            json={"text": "AB", "model_id": "libri"},
        )
        first = json.loads(await asyncio.wait_for(response.content.readline(), timeout=1))
        self.assertEqual(first["alignment"]["characters"], ["A"])
        self.release.set()
        second = json.loads(await response.content.readline())
        self.assertEqual(second["alignment"]["characters"], ["B"])
        self.assertEqual(await response.content.read(), b"")
        self.assertEqual(response.content_type, "text/event-stream")


if __name__ == "__main__":
    unittest.main()
