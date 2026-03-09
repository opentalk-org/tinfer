from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

@dataclass
class IntermediateRepresentation:
    data: Any
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)

class TTSModel(ABC):
    @abstractmethod
    def load(self, path: str, voices_folder: str | None = None) -> None:
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_from_state(self, state: dict[str, Any]) -> None:
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass

    @property
    @abstractmethod
    def supports_parallel_chunks(self) -> bool:
        pass

    @property
    @abstractmethod
    def supports_word_alignment(self) -> bool:
        pass

    @abstractmethod
    def generate_batch(
        self,
        texts: list[str],
        contexts: list[dict[str, Any] | None],
        params: list[dict[str, Any]],
        request_metadata: list[dict[str, Any]],
    ) -> list[IntermediateRepresentation]:
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        context: dict[str, Any] | None,
        params: dict[str, Any],
        request_metadata: dict[str, Any],
    ) -> IntermediateRepresentation:
        pass

    @abstractmethod
    async def stream_batch(
        self,
        texts: list[str],
        contexts: list[dict[str, Any] | None],
        params: list[dict[str, Any]],
        request_metadata: list[dict[str, Any]],
    ) -> AsyncIterator[IntermediateRepresentation]:
        pass
    
    @abstractmethod
    async def stream(
        self,
        text: str,
        context: dict[str, Any] | None,
        params: dict[str, Any],
        request_metadata: dict[str, Any],
    ) -> AsyncIterator[IntermediateRepresentation]:
        pass