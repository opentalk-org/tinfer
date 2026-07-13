from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Transport(Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"


class SpeechOutputFormat(Enum):
    MP3_22050_32 = "mp3_22050_32"
    MP3_24000_48 = "mp3_24000_48"
    MP3_44100_32 = "mp3_44100_32"
    MP3_44100_64 = "mp3_44100_64"
    MP3_44100_96 = "mp3_44100_96"
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"
    PCM_8000 = "pcm_8000"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_32000 = "pcm_32000"
    PCM_44100 = "pcm_44100"
    PCM_48000 = "pcm_48000"
    WAV_8000 = "wav_8000"
    WAV_16000 = "wav_16000"
    WAV_22050 = "wav_22050"
    WAV_24000 = "wav_24000"
    WAV_32000 = "wav_32000"
    WAV_44100 = "wav_44100"
    WAV_48000 = "wav_48000"
    ULAW_8000 = "ulaw_8000"
    ALAW_8000 = "alaw_8000"
    OPUS_48000_32 = "opus_48000_32"
    OPUS_48000_64 = "opus_48000_64"
    OPUS_48000_96 = "opus_48000_96"
    OPUS_48000_128 = "opus_48000_128"
    OPUS_48000_192 = "opus_48000_192"


@dataclass(frozen=True)
class SpeechQuery:
    model_id: str | None
    output_format: SpeechOutputFormat
    language_code: str | None
    sync_alignment: bool
    inactivity_timeout: float
    auto_mode: bool
    enable_logging: bool
    enable_ssml_parsing: bool
    apply_text_normalization: str
    seed: int | None
    optimize_streaming_latency: int | None
    authorization: str | None
    single_use_token: str | None


@dataclass(frozen=True)
class VoiceSettings:
    speed: float | None
    alpha: float | None
    beta: float | None
    stability: float | None
    similarity_boost: float | None
    style: float | None
    use_speaker_boost: bool | None


@dataclass(frozen=True)
class GenerationConfig:
    chunk_length_schedule: tuple[int, ...]


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    model_id: str | None
    language_code: str | None
    voice_settings: VoiceSettings
    generation_config: GenerationConfig
    seed: int | None
    use_pvc_as_ivc: bool | None
    apply_text_normalization: str | None
    apply_language_text_normalization: bool | None
