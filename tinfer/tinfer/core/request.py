import queue
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, TypedDict
import uuid

import numpy as np

from tinfer.utils.audio_encoder import AudioFormat


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
    target_encoding: AudioFormat | str | None
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

@dataclass
class AudioChunkIPC:
    request_id: str
    audio: dict[str, Any]
    sample_rate: int
    text_span: tuple[int, int]
    alignments: Alignment | None = None
    chunk_index: int = 0
    nonce: str = ""
    error: str | None = None
    context: dict[str, Any] | None = None

@dataclass
class TTSRequestIPC:
    request_id: str
    ipc_id: str
    model_id: str
    voice_id: str
    text: str
    context: dict[str, Any]
    params: dict[str, Any]
    method: str = "generate_batch"
    chunk_index: int = 0
    created_at: float = field(default_factory=monotonic)
    worker_queued_at: float | None = None
    alignment_type: AlignmentType = AlignmentType.WORD
    is_first: bool = False
    start_time: float = 0.0
    collected_time: float = 0.0
    text_span: tuple[int, int] = (0, 0)
    nonce: str = ""
    target_sample_rate: int | None = None
    target_encoding: AudioFormat | None = None
    first_audio_latency_started_at: float | None = None
    source_text: str | None = None


@dataclass(frozen=True)
class PreparedTextChunk:
    text: str
    text_span: tuple[int, int]


@dataclass(frozen=True)
class ChunkLimits:
    trigger: int
    no_split_limit: int


@dataclass
class TTSRequest:
    request_id: str
    model_id: str
    voice_id: str

    tts_params: dict[str, Any] = field(default_factory=dict)

    text_buffer: str = ""
    text_committed_pos: int = 0

    chunker_state: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)

    chunk_length_schedule: list[int] = field(default_factory=lambda: [120, 160, 250, 290])
    timeout_trigger_ms: float = 80.0
    generation_window_started_at: float | None = None
    prepared_chunks: list[PreparedTextChunk] = field(default_factory=list)
    created_at: float = field(default_factory=monotonic)
    first_text_at: float | None = None
    first_audio_at: float | None = None
    force_next_generation: bool = False

    audio_queue: queue.Queue[AudioChunk] = field(default_factory=queue.Queue)
    alignment_type: AlignmentType = AlignmentType.WORD
    text_processor: Any | None = None
    pronunciation_applier: Any | None = None

    pending_chunks: int = 0
    collected_time: float = 0.0
    start_time: float = 0.0

    nonce: str = str(uuid.uuid4())
    target_sample_rate: int | None = None
    target_encoding: AudioFormat | None = None

    def __post_init__(self) -> None:
        if not self.chunk_length_schedule:
            raise ValueError("chunk length schedule must not be empty")
        invalid_values = (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.chunk_length_schedule
        )
        if any(invalid_values):
            raise ValueError("chunk length schedule values must be positive integers")
        schedule_pairs = zip(self.chunk_length_schedule, self.chunk_length_schedule[1:])
        if any(left > right for left, right in schedule_pairs):
            raise ValueError("chunk length schedule must be non-decreasing")

    def append_text(self, text: str) -> bool:
        if not text.strip():
            return False

        starts_generation_window = not self.get_pending_text().strip()
        now = monotonic()
        if self.first_text_at is None:
            self.first_text_at = now
        if starts_generation_window:
            self.generation_window_started_at = now
        self.text_buffer += text
        return starts_generation_window

    def get_pending_text(self) -> str:
        return self.text_buffer[self.text_committed_pos:]
    
    def commit_text(self, length_chars: int) -> None:
        self.text_committed_pos += length_chars

    def get_chunk_limits(self, chunk_index: int) -> ChunkLimits:
        schedule_index = min(chunk_index, len(self.chunk_length_schedule) - 1)
        trigger = self.chunk_length_schedule[schedule_index]

        if schedule_index + 1 < len(self.chunk_length_schedule):
            no_split_limit = self.chunk_length_schedule[schedule_index + 1]
        elif len(self.chunk_length_schedule) > 1:
            previous = self.chunk_length_schedule[-2]
            no_split_limit = trigger + max(1, trigger - previous)
        else:
            no_split_limit = trigger + max(1, trigger // 3)

        return ChunkLimits(trigger=trigger, no_split_limit=no_split_limit)
    
    def should_trigger_now(self, now: float) -> bool:
        pending_text = self.get_pending_text()
        if not pending_text.strip():
            return False

        if self.force_next_generation:
            self.force_next_generation = False
            return True
        
        assert self.generation_window_started_at is not None, "pending text requires a generation window"
        elapsed_ms = (now - self.generation_window_started_at) * 1000.0
        if elapsed_ms >= self.timeout_trigger_ms:
            return True

        chunk_index = self.chunker_state["chunk_index"] if "chunk_index" in self.chunker_state else 0
        if len(pending_text) > self.get_chunk_limits(chunk_index).trigger:
            return True
        
        return False

    def get_state(self) -> dict[str, Any]:
        return self.stream_state

    def set_state(self, state: dict[str, Any]) -> None:
        self.stream_state = state
