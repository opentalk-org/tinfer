import unittest
import io
from types import SimpleNamespace

import numpy as np
from multidict import MultiDict
from pydub import AudioSegment

from tinfer.core.request import Alignment, AlignmentItem, AlignmentType, AudioChunk, ModelInfo
from tinfer.server.websocket.query_parser import (
    map_stream_params,
    parse_query,
    validate_language,
)
from tinfer.server.websocket.response_formatter import (
    encode_chunk,
    format_http_timing,
    format_model,
    format_ws_audio,
)
from tinfer.server.websocket.schemas import SpeechOutputFormat, Transport
from tinfer.server.websocket.speech_parser import parse_speech_request


class ContractTypesTest(unittest.TestCase):
    def test_requires_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Field required"):
            parse_speech_request({})

    def test_documented_optional_fields_and_null_settings_are_accepted(self) -> None:
        speech = parse_speech_request(
            {
                "text": "Hello",
                "voice_settings": None,
                "pronunciation_dictionary_locators": [
                    {
                        "pronunciation_dictionary_id": "dictionary",
                        "version_id": "version",
                    }
                ],
                "previous_text": "Before",
                "next_text": "After",
                "previous_request_ids": ["previous"],
                "next_request_ids": ["next"],
            }
        )
        self.assertEqual(speech.text, "Hello")
        self.assertIsNone(speech.voice_settings.speed)

    def test_query_defaults_and_rejects_permissive_format_aliases(self) -> None:
        query = parse_query(SimpleNamespace(query=MultiDict()), Transport.WEBSOCKET)
        self.assertIsNone(query.model_id)
        self.assertEqual(query.output_format, SpeechOutputFormat.MP3_44100_128)
        http_query = parse_query(SimpleNamespace(query=MultiDict()), Transport.HTTP)
        self.assertEqual(http_query.output_format, SpeechOutputFormat.MP3_44100_128)
        ws_tokens = parse_query(
            SimpleNamespace(
                query=MultiDict({"authorization": "token", "single_use_token": "once"})
            ),
            Transport.WEBSOCKET,
        )
        self.assertEqual(ws_tokens.authorization, "token")

        request = SimpleNamespace(query=MultiDict({"output_format": "pcm-whatever-24000"}))
        with self.assertRaisesRegex(ValueError, "output_format"):
            parse_query(request, Transport.WEBSOCKET)
        unknown = SimpleNamespace(query=MultiDict({"unknown": "1"}))
        with self.assertRaisesRegex(ValueError, "unsupported query parameter"):
            parse_query(unknown, Transport.HTTP)

    def test_http_only_formats_are_rejected_by_websocket(self) -> None:
        for value in ("mp3_24000_48", "pcm_32000", "pcm_48000"):
            request = SimpleNamespace(query=MultiDict({"output_format": value}))
            self.assertEqual(parse_query(request, Transport.HTTP).output_format.value, value)
            with self.assertRaisesRegex(ValueError, "output_format"):
                parse_query(request, Transport.WEBSOCKET)

        chunk = AudioChunk(audio=np.ones(24_000, dtype=np.float32) * 0.1, sample_rate=24_000)
        encoded = encode_chunk(chunk, SpeechOutputFormat.PCM_32000)
        self.assertEqual(len(encoded), 64_000)

        mp3 = encode_chunk(chunk, SpeechOutputFormat.MP3_24000_48)
        decoded = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        self.assertEqual(decoded.frame_rate, 24_000)
        self.assertGreater(len(decoded), 900)

    def test_validates_language_from_model_metadata(self) -> None:
        query = parse_query(
            SimpleNamespace(query=MultiDict({"model_id": "libri", "language_code": "pl"})),
            Transport.WEBSOCKET,
        )
        info = ModelInfo("libri", ("en-us",), "en-us")
        self.assertEqual(validate_language(query, info), "en-us")

    def test_maps_effective_and_accepted_voice_fields(self) -> None:
        query = parse_query(
            SimpleNamespace(
                query=MultiDict(
                    {"model_id": "libri", "output_format": "pcm_24000", "language_code": "en-us"}
                )
            ),
            Transport.WEBSOCKET,
        )
        speech = parse_speech_request(
            {
                "text": "Hello ",
                "voice_settings": {
                    "speed": 1.1,
                    "alpha": 0.2,
                    "beta": 0.7,
                    "stability": 0.4,
                    "similarity_boost": 0.8,
                    "style": 0.1,
                    "use_speaker_boost": True,
                },
                "generation_config": {"chunk_length_schedule": [80, 120]},
            }
        )
        params = map_stream_params(query, speech, AlignmentType.CHAR)
        self.assertEqual(
            params["tts_params"],
            {
                "speed": 1.1,
                "style_interpolation_factor": 0.4,
                "embedding_scale": 1.1,
                "apply_text_normalization": "auto",
                "alpha": 0.2,
                "beta": 0.7,
                "language": "en-us",
            },
        )
        self.assertEqual(params["chunk_length_schedule"], [80, 120])
        self.assertNotIn("target_encoding", params)
        self.assertEqual(params["target_sample_rate"], 24_000)

    def test_chunk_schedule_uses_supported_range(self) -> None:
        for value in (49, 501):
            with self.assertRaisesRegex(ValueError, "50 through 500"):
                parse_speech_request(
                    {
                        "text": " ",
                        "generation_config": {"chunk_length_schedule": [value]},
                    }
                )

    def test_documented_bounds_are_enforced(self) -> None:
        for seed in (-1, 4_294_967_296):
            with self.assertRaisesRegex(ValueError, "seed"):
                parse_speech_request({"text": "Hello", "seed": seed})
        with self.assertRaisesRegex(ValueError, "maximum of 3"):
            parse_speech_request(
                {"text": "Hello", "previous_request_ids": ["id"] * 4}
            )
        with self.assertRaisesRegex(ValueError, "speed"):
            parse_speech_request(
                {"text": "Hello", "voice_settings": {"speed": 1.21}}
            )
        for query in (
            {"seed": "4294967296"},
            {"inactivity_timeout": "181"},
            {"inactivity_timeout": "1.5"},
            {"optimize_streaming_latency": "5"},
        ):
            transport = (
                Transport.HTTP
                if "optimize_streaming_latency" in query
                else Transport.WEBSOCKET
            )
            with self.assertRaises(ValueError):
                parse_query(SimpleNamespace(query=MultiDict(query)), transport)

    def test_accepted_no_effect_fields_do_not_change_stream_params(self) -> None:
        query = parse_query(
            SimpleNamespace(
                query=MultiDict(
                    {
                        "output_format": "pcm_24000",
                        "enable_logging": "false",
                        "optimize_streaming_latency": "2",
                    }
                )
            ),
            Transport.HTTP,
        )
        speech = parse_speech_request(
            {
                "text": "Hello",
                "seed": 7,
                "use_pvc_as_ivc": True,
                "apply_text_normalization": "auto",
                "apply_language_text_normalization": False,
            }
        )
        params = map_stream_params(query, speech, AlignmentType.NONE)
        self.assertEqual(
            params["tts_params"],
            {"seed": 7, "apply_text_normalization": "auto"},
        )

    def test_auto_mode_disables_generation_buffer_timeout(self) -> None:
        query = parse_query(
            SimpleNamespace(query=MultiDict({"auto_mode": "true"})),
            Transport.WEBSOCKET,
        )
        speech = parse_speech_request({"text": "Hello"})
        params = map_stream_params(query, speech, AlignmentType.NONE)
        self.assertEqual(params["timeout_trigger_ms"], 0.0)

    def test_body_text_normalization_overrides_query_mode(self) -> None:
        query = parse_query(
            SimpleNamespace(query=MultiDict({"apply_text_normalization": "on"})),
            Transport.WEBSOCKET,
        )
        speech = parse_speech_request(
            {"text": "Hello", "apply_text_normalization": "off"}
        )
        params = map_stream_params(query, speech, AlignmentType.NONE)
        self.assertEqual(params["tts_params"]["apply_text_normalization"], "off")

    def test_formats_model_and_alignment_casings(self) -> None:
        info = ModelInfo("libri", ("pl", "en-us"), "en-us")
        self.assertEqual(
            format_model(info),
            {
                "model_id": "libri",
                "name": "libri",
                "can_do_text_to_speech": True,
                "languages": [
                    {"language_id": "en-us", "name": "en-us"},
                    {"language_id": "pl", "name": "pl"},
                ],
                "default_language": "en-us",
            },
        )

        chunk = AudioChunk(
            audio=np.array([0.0], dtype=np.float32),
            sample_rate=24_000,
            alignments=Alignment(
                [AlignmentItem("A", 0, 1, 250, 750)],
                AlignmentType.CHAR,
            ),
        )
        ws = format_ws_audio(chunk, SpeechOutputFormat.PCM_24000, is_final=False, context_id="one")
        self.assertEqual(ws["alignment"]["charStartTimesMs"], [250])
        self.assertEqual(ws["alignment"]["charDurationsMs"], [500])
        self.assertEqual(ws["contextId"], "one")
        self.assertFalse(ws["isFinal"])

        http = format_http_timing(chunk, SpeechOutputFormat.PCM_24000)
        self.assertEqual(http["alignment"]["character_start_times_seconds"], [0.25])
        self.assertEqual(http["alignment"]["character_end_times_seconds"], [0.75])


if __name__ == "__main__":
    unittest.main()
