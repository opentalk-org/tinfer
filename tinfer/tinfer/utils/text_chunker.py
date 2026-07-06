from __future__ import annotations

import re
from time import monotonic

import pysbd

from tinfer.core.request import TTSRequest


class TextChunker:
    def __init__(self, language: str = "pl") -> None:
        self._segmenter = pysbd.Segmenter(language=language, clean=False)

    def get_chunks(self, request: TTSRequest, single_chunk: bool = False):
        now = monotonic()
        pending_text = request.get_pending_text()

        if not request.should_trigger_now(now):
            return []

        current_chunk_index = request.chunker_state.get("chunk_index", 0)
        chunks = self.split_text_if_needed(pending_text, request, current_chunk_index)
        chunks = self._enforce_min_chars_trigger(chunks, request.min_chars_trigger)
        if not chunks:
            return []

        if single_chunk:
            chunk = chunks[0]
            text_span_start = request.text_committed_pos
            text_span_end = request.text_committed_pos + len(chunk)
            request.chunker_state["chunk_index"] = current_chunk_index + 1
            return [(chunk, current_chunk_index, len(chunks) == 1, (text_span_start, text_span_end))]

        results = []
        offset = 0
        for i, chunk in enumerate(chunks):
            text_span_start = request.text_committed_pos + offset
            text_span_end = text_span_start + len(chunk)
            results.append((chunk, current_chunk_index, i == len(chunks) - 1, (text_span_start, text_span_end)))
            offset += len(chunk)
            current_chunk_index += 1

        request.chunker_state["chunk_index"] = current_chunk_index
        return results

    def split_text_if_needed(self, text: str, request: TTSRequest, chunk_index: int) -> list[str]:
        if len(text) <= self.get_max_chunk_size(request, chunk_index):
            return [text]

        sentence_chunks = self._sentence_chunks(text)
        pieces: list[str] = []
        for sentence in sentence_chunks:
            pieces.extend(self._split_oversized(sentence, request, chunk_index + len(pieces)))
        return self._pack_to_schedule(pieces, request, chunk_index)

    def _enforce_min_chars_trigger(self, chunks: list[str], min_chars_trigger: int) -> list[str]:
        if min_chars_trigger <= 0:
            return chunks

        result = []
        pending = ""

        for chunk in chunks:
            if not chunk:
                continue

            if pending:
                pending += chunk
                if len(pending.strip()) >= min_chars_trigger:
                    result.append(pending)
                    pending = ""
                continue

            if len(chunk.strip()) >= min_chars_trigger:
                result.append(chunk)
            else:
                pending = chunk

        if pending:
            if result:
                result[-1] += pending
            elif len(pending.strip()) >= min_chars_trigger:
                result.append(pending)

        return result

    def get_max_chunk_size(self, request: TTSRequest, chunk_index: int) -> int:
        if chunk_index >= len(request.chunk_length_schedule):
            return request.chunk_length_schedule[-1]
        return request.chunk_length_schedule[chunk_index]

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
                max_size = self.get_max_chunk_size(request, chunk_index + len(next_pending))
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
            max_size = self.get_max_chunk_size(request, chunk_index + len(result))
            if len(part) <= max_size:
                result.append(part)
            else:
                result.extend(part[i : i + max_size] for i in range(0, len(part), max_size))
        return [chunk for chunk in result if chunk]

    def _pack_to_schedule(self, pieces: list[str], request: TTSRequest, chunk_index: int) -> list[str]:
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            if not piece:
                continue

            current_index = chunk_index + len(chunks)
            max_size = self.get_max_chunk_size(request, current_index)
            if current and len(current) + len(piece) > max_size:
                chunks.append(current)
                current = piece
            else:
                current += piece

        if current:
            chunks.append(current.rstrip())

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
