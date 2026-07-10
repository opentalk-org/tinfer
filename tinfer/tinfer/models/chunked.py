from __future__ import annotations
from abc import abstractmethod
from typing import Any
from typing import AsyncIterator

from tinfer.models.base.model import TTSModel, IntermediateRepresentation


class ChunkedModel(TTSModel):
    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    @abstractmethod
    def supports_parallel_sequential_chunks(self) -> bool:
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

    def generate(
        self,
        text: str,
        context: dict[str, Any] | None,
        params: dict[str, Any],
        request_metadata: dict[str, Any],
    ) -> IntermediateRepresentation:
        results = self.generate_batch(
            [text],
            [context],
            [params],
            [request_metadata],
        )
        return results[0]

    async def stream_batch(
        self,
        texts: list[str],
        contexts: list[dict[str, Any] | None],
        params: list[dict[str, Any]],
        request_metadata: list[dict[str, Any]],
    ) -> AsyncIterator[IntermediateRepresentation]:
        results = self.generate_batch(texts, contexts, params, request_metadata)
        for result in results:
            yield result
    
    async def stream(
        self,
        text: str,
        context: dict[str, Any] | None,
        params: dict[str, Any],
        request_metadata: dict[str, Any],
    ) -> AsyncIterator[IntermediateRepresentation]:
        result = self.generate(text, context, params, request_metadata)
        yield result