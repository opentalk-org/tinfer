import io
import unittest
import wave

from aiohttp.test_utils import TestClient, TestServer
from pydub import AudioSegment

from tinfer.server.websocket.server import WebSocketServer
from tmp_tests.tts_api.fakes import FakeTTS


class AudioFormatsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = TestClient(TestServer(WebSocketServer(FakeTTS()).create_app()))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_unary_wav_formats_have_requested_sample_rate(self) -> None:
        for sample_rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
            response = await self.client.post(
                f"/v1/text-to-speech/voice?output_format=wav_{sample_rate}",
                json={"text": "Hello"},
            )
            self.assertEqual(response.status, 200)
            with wave.open(io.BytesIO(await response.read()), "rb") as audio:
                self.assertEqual(audio.getframerate(), sample_rate)
                self.assertEqual(audio.getnchannels(), 1)

    async def test_compressed_stream_is_one_decodable_response(self) -> None:
        response = await self.client.post(
            "/v1/text-to-speech/voice/stream?output_format=mp3_44100_128",
            json={"text": "Hello"},
        )
        encoded = await response.read()
        decoded = AudioSegment.from_file(io.BytesIO(encoded), format="mp3")
        self.assertEqual(response.status, 200)
        self.assertGreater(len(decoded.raw_data), 4)


if __name__ == "__main__":
    unittest.main()
