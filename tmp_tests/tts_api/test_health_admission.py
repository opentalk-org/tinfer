import asyncio
import unittest

from aiohttp.test_utils import TestClient, TestServer

from tinfer.server.health import HealthState
from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeTTS


class HealthAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_counts_complete_transport_lifetimes(self) -> None:
        release = asyncio.Event()
        health = HealthState(warmup_complete=True)
        app = WebSocketServer(FakeTTS(release), health=health).create_app()
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            stream = await client.post(
                "/v1/text-to-speech/voice/stream?output_format=pcm_24000",
                json={"text": "AB", "model_id": "libri"},
            )
            await stream.content.readexactly(4)
            unary_task = asyncio.create_task(
                client.post(
                    "/v1/text-to-speech/voice?output_format=pcm_24000",
                    json={"text": "AB", "model_id": "libri"},
                )
            )
            for _ in range(20):
                if health.active_connections == 2:
                    break
                await asyncio.sleep(0.01)
            single = await client.ws_connect(
                "/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_24000"
            )
            multi = await client.ws_connect(
                "/v1/text-to-speech/voice/multi-stream-input?model_id=libri&output_format=pcm_24000"
            )
            self.assertEqual(health.active_connections, 4)

            await single.send_json({"text": ""})
            await single.receive_json()
            await single.receive()
            for _ in range(100):
                if health.active_connections == 3:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(health.active_connections, 3)
            await multi.send_json({"close_socket": True})
            await multi.receive()
            await multi.close()
            for _ in range(100):
                if health.active_connections == 2:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(health.active_connections, 2)
            release.set()
            unary = await unary_task
            await unary.read()
            await stream.read()
            for _ in range(100):
                if health.active_connections == 0:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(health.active_connections, 0)
        finally:
            release.set()
            await client.close()


if __name__ == "__main__":
    unittest.main()
