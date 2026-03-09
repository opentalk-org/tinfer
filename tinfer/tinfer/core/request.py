import uuid

from dataclasses import dataclass, field
from typing import Any, TypedDict
import queue
from enum import Enum
from time import monotonic

import numpy as np
from tinfer.utils.audio_encoder import AudioFormat


class AlignmentType(Enum):
    WORD = "word"
    CHAR = "char"
    PHONEME = "phoneme"
    NONE = "none"


class StreamParams(TypedDict, total=False):
    chunk_length_schedule: list[int]
    min_chunk_length_schedule: list[int]
    min_chars_trigger: int
    timeout_trigger_ms: float
    alignment_type: AlignmentType
    target_sample_rate: int | None
    target_encoding: AudioFormat | str | None
    tts_params: dict[str, Any]


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
    alignment_type: AlignmentType = AlignmentType.WORD
    is_first: bool = False
    start_time: float = 0.0
    collected_time: float = 0.0
    text_span: tuple[int, int] = (0, 0)
    nonce: str = ""
    target_sample_rate: int | None = None
    target_encoding: AudioFormat | None = None

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
    min_chunk_length_schedule: list[int] = field(default_factory=lambda: [50, 80, 120, 150])
    min_chars_trigger: int = 10
    timeout_trigger_ms: float = 80.0
    last_commit_time: float | None = None
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

    def append_text(self, text: str) -> None:
        if self.first_text_at is None:
            self.first_text_at = monotonic()
        self.text_buffer += text

    def get_pending_text(self) -> str:
        return self.text_buffer[self.text_committed_pos:]
    
    def commit_text(self, length_chars: int) -> None:
        self.text_committed_pos += length_chars
    
    def should_trigger_now(self, now: float) -> bool:
        pending_text = self.get_pending_text()
        if len(pending_text) < self.min_chars_trigger:
            return False

        if self.force_next_generation:
            self.force_next_generation = False
            return True
        
        reference_time = self.last_commit_time
        if reference_time is None:
            reference_time = self.first_text_at
        if reference_time is None:
            reference_time = self.created_at
        
        elapsed_ms = (now - reference_time) * 1000.0
        if elapsed_ms >= self.timeout_trigger_ms:
            return True

        if len(pending_text) > self.chunk_length_schedule[self.chunker_state.get("chunk_index", 0)]:
            return True
        
        return False

    def get_state(self) -> dict[str, Any]:
        return self.stream_state

    def set_state(self, state: dict[str, Any]) -> None:
        self.stream_state = state

    


