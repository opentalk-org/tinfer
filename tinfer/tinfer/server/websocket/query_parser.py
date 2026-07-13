from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tinfer.core.request import AlignmentType, ModelInfo, StreamParams
from tinfer.server.websocket.capability_mapper import map_styletts2_settings
from tinfer.server.websocket.schemas import (
    SpeechOutputFormat,
    SpeechQuery,
    SpeechRequest,
    Transport,
)

DEFAULT_OUTPUT_FORMAT = SpeechOutputFormat.MP3_44100_128
HTTP_ONLY_FORMATS = {
    SpeechOutputFormat.MP3_24000_48,
    SpeechOutputFormat.PCM_32000,
    SpeechOutputFormat.PCM_48000,
    SpeechOutputFormat.WAV_8000,
    SpeechOutputFormat.WAV_16000,
    SpeechOutputFormat.WAV_22050,
    SpeechOutputFormat.WAV_24000,
    SpeechOutputFormat.WAV_32000,
    SpeechOutputFormat.WAV_44100,
    SpeechOutputFormat.WAV_48000,
}
HTTP_QUERY_FIELDS = {"output_format", "enable_logging", "optimize_streaming_latency"}
WEBSOCKET_QUERY_FIELDS = {
    "model_id",
    "output_format",
    "language_code",
    "sync_alignment",
    "inactivity_timeout",
    "auto_mode",
    "enable_logging",
    "enable_ssml_parsing",
    "apply_text_normalization",
    "seed",
    "authorization",
    "single_use_token",
}


def parse_query(request: Any, transport: Transport) -> SpeechQuery:
    query = request.query
    allowed_fields = (
        HTTP_QUERY_FIELDS if transport is Transport.HTTP else WEBSOCKET_QUERY_FIELDS
    )
    unknown_fields = set(query) - allowed_fields
    if unknown_fields:
        raise ValueError(f"unsupported query parameter: {sorted(unknown_fields)[0]}")
    output_value = (
        query["output_format"]
        if "output_format" in query
        else DEFAULT_OUTPUT_FORMAT.value
    )
    try:
        output_format = SpeechOutputFormat(output_value)
    except ValueError as error:
        raise ValueError(f"unsupported output_format: {output_value}") from error
    if transport is Transport.WEBSOCKET and output_format in HTTP_ONLY_FORMATS:
        raise ValueError(f"unsupported output_format for WebSocket: {output_value}")
    normalization = (
        query["apply_text_normalization"]
        if "apply_text_normalization" in query
        else "auto"
    )
    if normalization not in ("auto", "on", "off"):
        raise ValueError("apply_text_normalization must be auto, on, or off")
    inactivity_timeout = _inactivity_timeout(query)
    return SpeechQuery(
        model_id=_optional_string(query, "model_id"),
        output_format=output_format,
        language_code=_optional_string(query, "language_code"),
        sync_alignment=_parse_bool(query, "sync_alignment", False),
        inactivity_timeout=inactivity_timeout,
        auto_mode=_parse_bool(query, "auto_mode", False),
        enable_logging=_parse_bool(query, "enable_logging", True),
        enable_ssml_parsing=_parse_bool(query, "enable_ssml_parsing", False),
        apply_text_normalization=normalization,
        seed=_bounded_int(query, "seed", 0, 4_294_967_295),
        optimize_streaming_latency=_bounded_int(
            query, "optimize_streaming_latency", 0, 4
        ),
        authorization=_optional_string(query, "authorization"),
        single_use_token=_optional_string(query, "single_use_token"),
    )


def validate_language(query: SpeechQuery, info: ModelInfo) -> str:
    language = query.language_code if query.language_code is not None else info.default_language
    if language not in info.supported_languages:
        return info.default_language
    return language


def map_stream_params(
    query: SpeechQuery,
    speech: SpeechRequest,
    alignment_type: AlignmentType,
) -> StreamParams:
    seed = speech.seed if speech.seed is not None else query.seed
    tts_params = map_styletts2_settings(speech.voice_settings, seed)
    tts_params["apply_text_normalization"] = (
        speech.apply_text_normalization
        if speech.apply_text_normalization is not None
        else query.apply_text_normalization
    )
    if query.language_code is not None:
        tts_params["language"] = query.language_code
    return {
        "chunk_length_schedule": list(speech.generation_config.chunk_length_schedule),
        "timeout_trigger_ms": 0.0 if query.auto_mode else 80.0,
        "alignment_type": alignment_type,
        "target_sample_rate": int(query.output_format.value.split("_")[1]),
        "tts_params": tts_params,
    }


def _inactivity_timeout(query: Mapping[str, Any]) -> float:
    value = query["inactivity_timeout"] if "inactivity_timeout" in query else "20"
    try:
        timeout = int(value)
    except ValueError as error:
        raise ValueError("inactivity_timeout must be an integer") from error
    if not 1 <= timeout <= 180:
        raise ValueError("inactivity_timeout must be between 1 and 180")
    return float(timeout)


def _parse_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{key} must be true or false")


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{key} must be an integer") from error
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _bounded_int(
    payload: Mapping[str, Any],
    key: str,
    minimum: int,
    maximum: int,
) -> int | None:
    value = _optional_int(payload, key)
    if value is not None and not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value
