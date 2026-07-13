from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tinfer.core.request import Alignment, AlignmentItem, AlignmentType, AudioChunk, ModelInfo


def audio_chunk(index: int, value: float = 0.25) -> AudioChunk:
    return AudioChunk(
        audio=np.array([value, -value], dtype=np.float32),
        sample_rate=24_000,
        chunk_index=index,
        alignments=Alignment(
            [AlignmentItem(chr(65 + index), index, index + 1, index * 100, (index + 1) * 100)],
            AlignmentType.CHAR,
        ),
    )


@dataclass
class FakeStream:
    chunks: list[AudioChunk]
    pause_after_first: asyncio.Event | None = None
    texts: list[str] = field(default_factory=list)
    force_count: int = 0
    try_count: int = 0
    close_count: int = 0
    pull_count: int = 0

    def add_text(self, text: str) -> None:
        self.texts.append(text)

    def force_generate(self) -> None:
        self.force_count += 1

    def try_generate(self) -> None:
        self.try_count += 1

    async def pull_audio(self):
        self.pull_count += 1
        if self.pull_count > 1:
            return
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if index == 0 and self.pause_after_first is not None:
                await self.pause_after_first.wait()

    def close(self) -> None:
        self.close_count += 1


class FakeTTS:
    def __init__(self, pause_after_first: asyncio.Event | None = None) -> None:
        self.model_infos = [ModelInfo("libri", ("en-us", "en"), "en-us")]
        self.pause_after_first = pause_after_first
        self.streams: list[FakeStream] = []
        self.params: list[dict[str, Any]] = []

    def get_model_infos(self) -> list[ModelInfo]:
        return self.model_infos

    def get_model_ids(self) -> list[str]:
        return [info.model_id for info in self.model_infos]

    def get_voice_ids(self, model_id: str) -> list[str]:
        if model_id != "libri":
            raise ValueError(f"unknown model: {model_id}")
        return ["voice"]

    def create_stream(self, model_id: str, voice_id: str, params: dict[str, Any]) -> FakeStream:
        if model_id != "libri" or voice_id != "voice":
            raise ValueError("unknown model or voice")
        stream = FakeStream([audio_chunk(0), audio_chunk(1)], self.pause_after_first)
        self.streams.append(stream)
        self.params.append(params)
        return stream


@dataclass
class PumpFailureStream(FakeStream):
    async def pull_audio(self):
        raise RuntimeError("audio pump failed")
        yield audio_chunk(0)


@dataclass
class DelayedCleanupStream(FakeStream):
    pull_started: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_started: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_release: asyncio.Event = field(default_factory=asyncio.Event)

    async def pull_audio(self):
        self.pull_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cleanup_started.set()
            await self.cleanup_release.wait()
            raise
        yield audio_chunk(0)


class DelayedCleanupTTS(FakeTTS):
    def create_stream(self, model_id: str, voice_id: str, params: dict[str, Any]) -> FakeStream:
        if model_id != "libri" or voice_id != "voice":
            raise ValueError("unknown model or voice")
        stream = DelayedCleanupStream([])
        self.streams.append(stream)
        self.params.append(params)
        return stream
