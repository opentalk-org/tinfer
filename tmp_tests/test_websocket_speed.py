import unittest
import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import types


def _install_handler_import_stubs():
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web_ws = types.ModuleType("aiohttp.web_ws")
    aiohttp.web_ws.WebSocketResponse = object

    async_engine = types.ModuleType("tinfer.core.async_engine")
    async_engine.AsyncStreamingTTS = object

    request = types.ModuleType("tinfer.core.request")

    class AlignmentType:
        CHAR = "char"
        NONE = "none"

    request.AudioChunk = object
    request.AlignmentType = AlignmentType

    latency = types.ModuleType("tinfer.support.latency")

    class FirstAudioLatencyTimer:
        def start(self):
            pass

        def consume_ms(self):
            return None

    latency.FirstAudioLatencyTimer = FirstAudioLatencyTimer

    observability = types.ModuleType("tinfer.support.observability")

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    @contextmanager
    def start_span(*args, **kwargs):
        yield None

    observability.get_logger = lambda name: Logger()
    observability.start_span = start_span

    alignment_formatter = types.ModuleType("tinfer.utils.alignment_formatter")
    alignment_formatter.AlignmentFormatter = object

    audio_encoder = types.ModuleType("tinfer.utils.audio_encoder")
    audio_encoder.encode_audio_to_base64 = lambda audio, sample_rate, output_format: ""
    audio_encoder.parse_output_format = lambda output_format: output_format
    audio_encoder.get_sample_rate = lambda output_format: 24000

    sys.modules["aiohttp"] = aiohttp
    sys.modules["aiohttp.web_ws"] = aiohttp.web_ws
    sys.modules["tinfer.core.async_engine"] = async_engine
    sys.modules["tinfer.core.request"] = request
    sys.modules["tinfer.support.latency"] = latency
    sys.modules["tinfer.support.observability"] = observability
    sys.modules["tinfer.utils.alignment_formatter"] = alignment_formatter
    sys.modules["tinfer.utils.audio_encoder"] = audio_encoder


def _load_handler_class():
    _install_handler_import_stubs()
    handler_path = (
        Path(__file__).resolve().parents[1]
        / "tinfer"
        / "tinfer"
        / "server"
        / "websocket"
        / "handler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "websocket_handler_under_test",
        handler_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebSocketHandler


WebSocketHandler = _load_handler_class()


class WebSocketSpeedParamTests(unittest.TestCase):
    def test_maps_voice_settings_speed_to_tts_params(self):
        handler = WebSocketHandler(
            ws=None,
            tts=None,
            voice_id="any",
            query_params={"output_format": "pcm_24000"},
        )

        params = handler._map_params_to_tinfer(
            voice_settings={"speed": 1.25},
            generation_config={},
        )

        self.assertIn("tts_params", params)
        self.assertEqual(params["tts_params"], {"speed": 1.25})


if __name__ == "__main__":
    unittest.main()
