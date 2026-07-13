import unittest

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from tinfer.core.request import AudioChunk
from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeStream, FakeTTS


class FailedTTS(FakeTTS):
    def create_stream(self, model_id, voice_id, params):
        stream = FakeStream(
            [AudioChunk(np.array([], dtype=np.float32), 24_000, error="inference failed")]
        )
        self.streams.append(stream)
        return stream


class ErrorContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = TestClient(TestServer(WebSocketServer(FakeTTS()).create_app()))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_missing_text_uses_validation_issue_shape(self) -> None:
        response = await self.client.post("/v1/text-to-speech/voice", json={})
        self.assertEqual(response.status, 422)
        self.assertEqual(
            await response.json(),
            {
                "detail": [
                    {
                        "loc": ["body", "text"],
                        "msg": "Field required",
                        "type": "missing",
                    }
                ]
            },
        )

    async def test_inference_error_does_not_return_empty_success(self) -> None:
        client = TestClient(TestServer(WebSocketServer(FailedTTS()).create_app()))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/text-to-speech/voice?output_format=pcm_24000",
                json={"text": "Hello"},
            )
            self.assertEqual(response.status, 500)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
