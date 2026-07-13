from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tinfer.server.websocket.errors import RequestValidationError, ValidationIssue
from tinfer.server.websocket.schemas import (
    GenerationConfig,
    SpeechRequest,
    VoiceSettings,
)

DEFAULT_CHUNK_SCHEDULE = (120, 160, 250, 290)
SPEECH_FIELDS = {
    "text",
    "model_id",
    "language_code",
    "voice_settings",
    "generation_config",
    "seed",
    "use_pvc_as_ivc",
    "apply_text_normalization",
    "apply_language_text_normalization",
    "pronunciation_dictionary_locators",
    "previous_text",
    "next_text",
    "previous_request_ids",
    "next_request_ids",
}
VOICE_FIELDS = {
    "speed",
    "alpha",
    "beta",
    "stability",
    "similarity_boost",
    "style",
    "use_speaker_boost",
}
GENERATION_FIELDS = {"chunk_length_schedule"}


def parse_speech_request(payload: Mapping[str, Any]) -> SpeechRequest:
    unknown_fields = set(payload) - SPEECH_FIELDS
    if unknown_fields:
        raise ValueError(f"unsupported request field: {sorted(unknown_fields)[0]}")
    if "text" not in payload:
        raise RequestValidationError(
            ValidationIssue(("body", "text"), "Field required", "missing")
        )
    if not isinstance(payload["text"], str):
        raise RequestValidationError(
            ValidationIssue(
                ("body", "text"),
                "Input should be a valid string",
                "string_type",
            )
        )
    voice_payload = _optional_mapping(payload, "voice_settings")
    generation_payload = _optional_mapping(payload, "generation_config")
    unknown_voice_fields = set(voice_payload) - VOICE_FIELDS
    if unknown_voice_fields:
        raise ValueError(f"unsupported voice_settings field: {sorted(unknown_voice_fields)[0]}")
    unknown_generation_fields = set(generation_payload) - GENERATION_FIELDS
    if unknown_generation_fields:
        raise ValueError(
            f"unsupported generation_config field: {sorted(unknown_generation_fields)[0]}"
        )
    _validate_dictionary_locators(payload)
    _validate_optional_string(payload, "previous_text", allow_empty=True)
    _validate_optional_string(payload, "next_text", allow_empty=True)
    _validate_string_sequence(payload, "previous_request_ids")
    _validate_string_sequence(payload, "next_request_ids")
    schedule = _chunk_schedule(generation_payload)
    return SpeechRequest(
        text=payload["text"],
        model_id=_optional_string(payload, "model_id"),
        language_code=_optional_string(payload, "language_code"),
        voice_settings=_voice_settings(voice_payload),
        generation_config=GenerationConfig(schedule),
        seed=_bounded_int(payload, "seed", 0, 4_294_967_295),
        use_pvc_as_ivc=_optional_bool(payload, "use_pvc_as_ivc"),
        apply_text_normalization=_optional_normalization(payload, "apply_text_normalization"),
        apply_language_text_normalization=_optional_bool(
            payload, "apply_language_text_normalization"
        ),
    )


def _voice_settings(payload: Mapping[str, Any]) -> VoiceSettings:
    return VoiceSettings(
        speed=_bounded_float(payload, "speed", 0.7, 1.2),
        alpha=_bounded_float(payload, "alpha", 0.0, 1.0),
        beta=_bounded_float(payload, "beta", 0.0, 1.0),
        stability=_bounded_float(payload, "stability", 0.0, 1.0),
        similarity_boost=_bounded_float(payload, "similarity_boost", 0.0, 1.0),
        style=_bounded_float(payload, "style", 0.0, 1.0),
        use_speaker_boost=_optional_bool(payload, "use_speaker_boost"),
    )


def _chunk_schedule(payload: Mapping[str, Any]) -> tuple[int, ...]:
    value = payload["chunk_length_schedule"] if "chunk_length_schedule" in payload else DEFAULT_CHUNK_SCHEDULE
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("chunk_length_schedule must be a non-empty array")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError("chunk_length_schedule values must be integers")
    if any(item < 50 or item > 500 for item in value):
        raise ValueError("chunk_length_schedule values must be 50 through 500")
    return tuple(value)


def _validate_dictionary_locators(payload: Mapping[str, Any]) -> None:
    value = payload["pronunciation_dictionary_locators"] if "pronunciation_dictionary_locators" in payload else None
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise ValueError("pronunciation_dictionary_locators must be an array")
    if len(value) > 3:
        raise ValueError("pronunciation_dictionary_locators accepts a maximum of 3 values")
    for locator in value:
        if not isinstance(locator, Mapping):
            raise ValueError("pronunciation_dictionary_locators values must be objects")
        if set(locator) - {"pronunciation_dictionary_id", "version_id"}:
            raise ValueError("unsupported pronunciation_dictionary_locators field")
        dictionary_id = _optional_string(locator, "pronunciation_dictionary_id")
        if dictionary_id is None:
            raise ValueError("pronunciation_dictionary_id is required")
        _optional_string(locator, "version_id")


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in payload or payload[key] is None:
        return {}
    if not isinstance(payload[key], Mapping):
        raise ValueError(f"{key} must be an object")
    return payload[key]


def _validate_string_sequence(payload: Mapping[str, Any], key: str) -> None:
    if key not in payload or payload[key] is None:
        return
    value = payload[key]
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    if len(value) > 3:
        raise ValueError(f"{key} accepts a maximum of 3 values")


def _validate_optional_string(
    payload: Mapping[str, Any], key: str, allow_empty: bool = False
) -> None:
    _optional_string(payload, key, allow_empty)


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _bounded_float(
    payload: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float,
) -> float | None:
    value = _optional_float(payload, key)
    if value is not None and not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    if key not in payload or payload[key] is None:
        return None
    if not isinstance(payload[key], bool):
        raise ValueError(f"{key} must be boolean")
    return payload[key]


def _optional_string(
    payload: Mapping[str, Any], key: str, allow_empty: bool = False
) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
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


def _optional_normalization(payload: Mapping[str, Any], key: str) -> str | None:
    value = _optional_string(payload, key)
    if value is not None and value not in ("auto", "on", "off"):
        raise ValueError(f"{key} must be auto, on, or off")
    return value
