import unittest

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from tinfer.server.health import HealthState
from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeTTS


class HttpSpeechTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tts = FakeTTS()
        self.health = HealthState(warmup_complete=True)
        app = WebSocketServer(self.tts, health=self.health).create_app()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_unary_audio_encodes_merged_raw_samples_once(self) -> None:
        response = await self.client.post(
            "/v1/text-to-speech/voice?output_format=pcm_24000",
            json={"text": "Hello ", "model_id": "libri", "language_code": "en-us"},
        )
        self.assertEqual(response.status, 200)
        expected = (np.array([0.25, -0.25, 0.25, -0.25]) * 32767).astype(np.int16).tobytes()
        self.assertEqual(await response.read(), expected)
        self.assertEqual(response.content_type, "application/octet-stream")
        self.assertEqual(self.tts.streams[0].close_count, 1)
        self.assertEqual(self.health.active_connections, 0)
        self.assertNotIn("target_encoding", self.tts.params[0])

    async def test_unary_timing_uses_seconds(self) -> None:
        response = await self.client.post(
            "/v1/text-to-speech/voice/with-timestamps?output_format=pcm_24000",
            json={"text": "AB", "model_id": "libri"},
        )
        payload = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["alignment"]["characters"], ["A", "B"])
        self.assertEqual(payload["alignment"]["character_start_times_seconds"], [0.0, 0.1])
        self.assertEqual(payload["alignment"]["character_end_times_seconds"], [0.1, 0.2])

    async def test_explicit_errors_precede_generation(self) -> None:
        malformed = await self.client.post("/v1/text-to-speech/voice", json={"model_id": "libri"})
        self.assertEqual(malformed.status, 422)

        missing = await self.client.post(
            "/v1/text-to-speech/missing", json={"text": "Hello", "model_id": "libri"}
        )
        self.assertEqual(missing.status, 404)

        await self.health.begin_draining()
        draining = await self.client.post(
            "/v1/text-to-speech/voice", json={"text": "Hello", "model_id": "libri"}
        )
        self.assertEqual(draining.status, 503)


if __name__ == "__main__":
    unittest.main()
