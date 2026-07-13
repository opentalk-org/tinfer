from __future__ import annotations

import re
from time import monotonic

import pysbd

from tinfer.core.request import PreparedTextChunk, TTSRequest


class TextChunker:
    def __init__(self, language: str = "pl") -> None:
        self._segmenter = pysbd.Segmenter(language=language, clean=False)

    def get_chunks(self, request: TTSRequest, single_chunk: bool = False):
        current_chunk_index = request.chunker_state.get("chunk_index", 0)
        if single_chunk and request.prepared_chunks:
            prepared = request.prepared_chunks.pop(0)
            if not prepared.text.strip():
                raise ValueError("text chunker produced an empty chunk")
            request.chunker_state["chunk_index"] = current_chunk_index + 1
            return [(prepared.text, current_chunk_index, not request.prepared_chunks, prepared.text_span)]

        if not request.should_trigger_now(monotonic()):
            return []

        pending_text = request.get_pending_text()
        chunks = self.split_text_if_needed(pending_text, request, current_chunk_index)
        if not chunks or any(not chunk.strip() for chunk in chunks):
            raise ValueError("text chunker produced an empty chunk")
        prepared_chunks = self._prepare_source_spans(pending_text, chunks, request.text_committed_pos)

        if single_chunk:
            prepared = prepared_chunks[0]
            request.prepared_chunks.extend(prepared_chunks[1:])
            request.chunker_state["chunk_index"] = current_chunk_index + 1
            return [(prepared.text, current_chunk_index, not request.prepared_chunks, prepared.text_span)]

        results = []
        for i, prepared in enumerate(prepared_chunks):
            results.append(
                (prepared.text, current_chunk_index, i == len(prepared_chunks) - 1, prepared.text_span)
            )
            current_chunk_index += 1

        request.chunker_state["chunk_index"] = current_chunk_index
        return results

    def split_text_if_needed(self, text: str, request: TTSRequest, chunk_index: int) -> list[str]:
        if len(text) <= request.get_chunk_limits(chunk_index).no_split_limit:
            return [text]

        sentence_chunks = self._sentence_chunks(text)
        pieces: list[str] = []
        for sentence in sentence_chunks:
            pieces.extend(self._split_oversized(sentence, request, chunk_index + len(pieces)))
        return self._pack_to_schedule(pieces, request, chunk_index)

    def _prepare_source_spans(
        self,
        text: str,
        chunks: list[str],
        text_offset: int,
    ) -> list[PreparedTextChunk]:
        prepared: list[PreparedTextChunk] = []
        cursor = 0

        for raw_chunk in chunks:
            chunk = raw_chunk
            start = text.find(chunk, cursor)
            if start < 0:
                raise ValueError("text chunker output does not match source text")
            end = start + len(chunk)
            if prepared:
                previous = prepared[-1]
                prepared[-1] = PreparedTextChunk(
                    text=previous.text,
                    text_span=(previous.text_span[0], text_offset + start),
                )
            prepared.append(
                PreparedTextChunk(
                    text=chunk,
                    text_span=(text_offset + start, text_offset + end),
                )
            )
            cursor = end

        if not prepared:
            raise ValueError("text chunker produced an empty chunk")
        last = prepared[-1]
        prepared[-1] = PreparedTextChunk(
            text=last.text,
            text_span=(last.text_span[0], text_offset + len(text)),
        )
        return prepared

    def get_target_chunk_size(self, request: TTSRequest, chunk_index: int) -> int:
        return request.get_chunk_limits(chunk_index).trigger

    def _sentence_chunks(self, text: str) -> list[str]:
        chunks: list[str] = []
        cursor = 0

        for sentence in self._segmenter.segment(text):
            start = text.find(sentence, cursor)
            if start < 0:
                return [text]
            if start > cursor:
                chunks.append(text[cursor:start])
            end = start + len(sentence)
            chunks.append(text[start:end])
            cursor = end

        if cursor < len(text):
            chunks.append(text[cursor:])

        return [chunk for chunk in chunks if chunk]

    def _split_oversized(self, text: str, request: TTSRequest, chunk_index: int) -> list[str]:
        pending = [text]
        for separator in ("\n\n", "\n", r"(?<=[.!?]) +", r"(?<=[,;]) +", " "):
            next_pending: list[str] = []
            changed = False
            for part in pending:
                max_size = self.get_target_chunk_size(request, chunk_index + len(next_pending))
                if len(part) <= max_size:
                    next_pending.append(part)
                    continue
                split_parts = self._split_keep_separator(part, separator)
                if len(split_parts) == 1:
                    next_pending.append(part)
                else:
                    changed = True
                    next_pending.extend(split_parts)
            pending = next_pending
            if changed:
                pending = self._pack_to_schedule(pending, request, chunk_index)

        result: list[str] = []
        for part in pending:
            max_size = self.get_target_chunk_size(request, chunk_index + len(result))
            if len(part) <= max_size:
                result.append(part)
            else:
                result.extend(self._split_by_derived_limits(part, request, chunk_index + len(result)))
        return [chunk for chunk in result if chunk]

    def _split_by_derived_limits(self, text: str, request: TTSRequest, chunk_index: int) -> list[str]:
        chunks: list[str] = []
        remaining = text

        while remaining:
            limits = request.get_chunk_limits(chunk_index + len(chunks))
            if len(remaining) <= limits.no_split_limit:
                chunks.append(remaining)
                break
            chunks.append(remaining[:limits.trigger])
            remaining = remaining[limits.trigger:]

        return chunks

    def _pack_to_schedule(self, pieces: list[str], request: TTSRequest, chunk_index: int) -> list[str]:
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            if not piece:
                continue

            current_index = chunk_index + len(chunks)
            max_size = self.get_target_chunk_size(request, current_index)
            if current and len(current) + len(piece) > max_size:
                chunks.append(current)
                current = piece
            else:
                current += piece

        if current:
            chunks.append(current)

        if len(chunks) > 1:
            previous_index = chunk_index + len(chunks) - 2
            no_split_limit = request.get_chunk_limits(previous_index).no_split_limit
            if len(chunks[-2]) + len(chunks[-1]) <= no_split_limit:
                chunks[-2] += chunks.pop()

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_keep_separator(self, text: str, separator: str) -> list[str]:
        if separator == " ":
            return [part for part in re.findall(r"\S+\s*", text) if part]

        if separator in ("\n\n", "\n"):
            parts = text.split(separator)
            result = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    part += separator
                if part:
                    result.append(part)
            return result

        parts = re.split(f"({separator})", text)
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if not part:
                i += 1
                continue
            if i + 1 < len(parts):
                part += parts[i + 1]
                i += 2
            else:
                i += 1
            result.append(part)
        return result

    def find_sentence_boundary(self, text: str, start_pos: int) -> int:
        if start_pos >= len(text):
            return len(text)

        match = re.search(r"[.!?]\s+", text[start_pos:])
        if match:
            return start_pos + match.start()
        return len(text)

    def find_punctuation_boundary(self, text: str, start_pos: int) -> int:
        if start_pos >= len(text):
            return len(text)

        match = re.search(r"[.,;:!?]\s*", text[start_pos:])
        if match:
            return start_pos + match.start()
        return len(text)
