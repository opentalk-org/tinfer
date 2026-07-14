from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

import numpy as np


class AlignmentType(Enum):
    WORD = "word"
    CHAR = "char"
    PHONEME = "phoneme"
    NONE = "none"


class StreamParams(TypedDict, total=False):
    chunk_length_schedule: list[int]
    timeout_trigger_ms: float
    alignment_type: AlignmentType
    target_sample_rate: int | None
    target_encoding: str | None
    tts_params: dict[str, Any]


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    supported_languages: tuple[str, ...]
    default_language: str


@dataclass(frozen=True)
class VoiceInfo:
    voice_id: str
    name: str
    category: str
    model_id: str


@dataclass
class AlignmentItem:
    item: str
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int


@dataclass
class Alignment:
    items: list[AlignmentItem] = field(default_factory=list)
    type_: AlignmentType = AlignmentType.WORD


@dataclass
class AudioChunk:
    audio: np.ndarray
    sample_rate: int
    chunk_index: int = 0
    text_span: tuple[int, int] = (0, 0)
    alignments: Alignment | None = None
    request_id: str = ""
    error: str | None = None


def chunk_from_native(chunk) -> AudioChunk:
    items = [AlignmentItem(*item) for item in chunk.alignment]
    return AudioChunk(
        audio=np.asarray(chunk.audio, dtype=np.float32),
        sample_rate=chunk.sample_rate,
        chunk_index=chunk.chunk_index,
        text_span=chunk.text_span,
        alignments=Alignment(items) if items else None,
    )
