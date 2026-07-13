from __future__ import annotations

import base64
import io
import wave
from typing import Any

import librosa
import numpy as np
from pydub import AudioSegment

from tinfer.core.request import Alignment, AudioChunk, ModelInfo, VoiceInfo
from tinfer.server.websocket.schemas import SpeechOutputFormat
from tinfer.utils.audio_encoder import DefaultAudioEncoder

_ENCODER = DefaultAudioEncoder()


def format_model(info: ModelInfo) -> dict[str, Any]:
    ordered = (info.default_language,) + tuple(
        language for language in info.supported_languages if language != info.default_language
    )
    return {
        "model_id": info.model_id,
        "name": info.model_id,
        "can_do_text_to_speech": True,
        "languages": [
            {"language_id": language, "name": language} for language in ordered
        ],
        "default_language": info.default_language,
    }


def format_voice(info: VoiceInfo) -> dict[str, Any]:
    return {
        "voice_id": info.voice_id,
        "name": info.name,
        "category": info.category,
        "labels": {},
        "model_id": info.model_id,
    }


def encode_chunk(chunk: AudioChunk, output_format: SpeechOutputFormat) -> bytes:
    if chunk.error is not None:
        raise RuntimeError(chunk.error)
    if output_format in (SpeechOutputFormat.PCM_32000, SpeechOutputFormat.PCM_48000):
        sample_rate = int(output_format.value.split("_")[1])
        audio = librosa.resample(chunk.audio, orig_sr=chunk.sample_rate, target_sr=sample_rate)
        return (audio * 32767.0).astype(np.int16).tobytes()
    if output_format == SpeechOutputFormat.MP3_24000_48:
        audio = librosa.resample(chunk.audio, orig_sr=chunk.sample_rate, target_sr=24_000)
        segment = AudioSegment(
            (audio * 32767.0).astype(np.int16).tobytes(),
            frame_rate=24_000,
            channels=1,
            sample_width=2,
        )
        buffer = io.BytesIO()
        segment.export(buffer, format="mp3", bitrate="48k")
        return buffer.getvalue()
    if output_format.value.startswith("wav_"):
        sample_rate = int(output_format.value.split("_")[1])
        audio = librosa.resample(
            chunk.audio,
            orig_sr=chunk.sample_rate,
            target_sr=sample_rate,
        )
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes((audio * 32767.0).astype(np.int16).tobytes())
        return buffer.getvalue()
    return _ENCODER.encode(chunk.audio, chunk.sample_rate, output_format)


def format_ws_audio(
    chunk: AudioChunk,
    output_format: SpeechOutputFormat,
    is_final: bool,
    context_id: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "audio": base64.b64encode(encode_chunk(chunk, output_format)).decode("ascii"),
        "isFinal": is_final,
    }
    if chunk.alignments is not None and chunk.alignments.items:
        alignment = _format_ws_alignment(chunk.alignments)
        response["alignment"] = alignment
        response["normalizedAlignment"] = alignment
    if context_id is not None:
        response["contextId"] = context_id
    return response


def format_http_timing(chunk: AudioChunk, output_format: SpeechOutputFormat) -> dict[str, Any]:
    alignment = _format_http_alignment(chunk.alignments)
    return {
        "audio_base64": base64.b64encode(encode_chunk(chunk, output_format)).decode("ascii"),
        "alignment": alignment,
        "normalized_alignment": alignment,
    }


def _format_ws_alignment(alignment: Alignment) -> dict[str, list[Any]]:
    return {
        "chars": [item.item for item in alignment.items],
        "charStartTimesMs": [item.start_ms for item in alignment.items],
        "charDurationsMs": [item.end_ms - item.start_ms for item in alignment.items],
    }


def _format_http_alignment(alignment: Alignment | None) -> dict[str, list[Any]]:
    items = alignment.items if alignment is not None else []
    return {
        "characters": [item.item for item in items],
        "character_start_times_seconds": [item.start_ms / 1000 for item in items],
        "character_end_times_seconds": [item.end_ms / 1000 for item in items],
    }
