import unittest

from aiohttp.client_exceptions import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeTTS


class ModelsApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = WebSocketServer(FakeTTS())
        self.app = self.server.create_app()
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_models_are_a_top_level_array_with_default_language_first(self) -> None:
        response = await self.client.get("/v1/models")
        payload = await response.json()
        self.assertIsInstance(payload, list)
        self.assertEqual(
            payload[0],
            {
                "model_id": "libri",
                "name": "libri",
                "can_do_text_to_speech": True,
                "languages": [
                    {"language_id": "en-us", "name": "en-us"},
                    {"language_id": "en", "name": "en"},
                ],
                "default_language": "en-us",
            },
        )

    async def test_voices_use_standard_envelope_and_identity_fields(self) -> None:
        response = await self.client.get("/v1/voices")
        self.assertEqual(response.status, 200)
        self.assertEqual(
            await response.json(),
            {
                "voices": [
                    {
                        "voice_id": "voice",
                        "name": "voice",
                        "category": "generated",
                        "labels": {},
                        "model_id": "libri",
                    }
                ]
            },
        )

    async def test_synthesis_routes_are_unique(self) -> None:
        routes = [(route.method, route.resource.canonical) for route in self.app.router.routes()]
        expected = {
            ("POST", "/v1/text-to-speech/{voice_id}"),
            ("POST", "/v1/text-to-speech/{voice_id}/with-timestamps"),
            ("POST", "/v1/text-to-speech/{voice_id}/stream"),
            ("POST", "/v1/text-to-speech/{voice_id}/stream/with-timestamps"),
            ("GET", "/v1/text-to-speech/{voice_id}/stream-input"),
            ("GET", "/v1/text-to-speech/{voice_id}/multi-stream-input"),
        }
        for route in expected:
            self.assertEqual(routes.count(route), 1)

    async def test_omitted_model_uses_loaded_default(self) -> None:
        response = await self.client.post(
            "/v1/text-to-speech/voice?output_format=pcm_24000",
            json={"text": "Hello"},
        )
        self.assertEqual(response.status, 200)
        self.assertGreater(len(await response.read()), 0)

    async def test_websocket_rejects_unknown_catalog_entries_before_upgrade(self) -> None:
        for path, status in (
            ("/v1/text-to-speech/voice/stream-input?model_id=missing", 404),
            ("/v1/text-to-speech/missing/stream-input?model_id=libri", 404),
            ("/v1/text-to-speech/voice/stream-input?model_id=libri&output_format=pcm_32000", 422),
        ):
            with self.assertRaises(WSServerHandshakeError) as raised:
                await self.client.ws_connect(path)
            self.assertEqual(raised.exception.status, status)


if __name__ == "__main__":
    unittest.main()
