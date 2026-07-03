from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import pysbd

from tinfer.core.request import TTSRequest

BASELINE_MIN_SENTENCE_SPLIT = 120


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str

    def text_from(self, source: str) -> str:
        return self.text


class TextChunker:
    def __init__(self, language: str = "pl", min_sentence_split: int = BASELINE_MIN_SENTENCE_SPLIT) -> None:
        self._segmenter = pysbd.Segmenter(language=language, clean=False)
        self._min_sentence_split = min_sentence_split

    def get_chunks(self, request: TTSRequest, single_chunk: bool = False):
        now = monotonic()
        pending_text = request.get_pending_text()

        if not request.should_trigger_now(now):
            return []

        current_chunk_index = request.chunker_state["chunk_index"] if "chunk_index" in request.chunker_state else 0
        spans = self.split_text_if_needed(pending_text, request, request.text_committed_pos)
        if not spans:
            return []

        if single_chunk:
            chunk_span = spans[0]
            request.chunker_state["chunk_index"] = current_chunk_index + 1
            return [
                (
                    chunk_span.text_from(request.text_buffer),
                    current_chunk_index,
                    len(spans) == 1,
                    (chunk_span.start, chunk_span.end),
                )
            ]

        results = []
        for span in spans:
            results.append(
                (
                    span.text_from(request.text_buffer),
                    current_chunk_index,
                    span == spans[-1],
                    (span.start, span.end),
                )
            )
            current_chunk_index += 1

        request.chunker_state["chunk_index"] = current_chunk_index
        return results

    def split_text_if_needed(self, text: str, request: TTSRequest, text_offset: int = 0) -> list[TextSpan]:
        if "||" in text:
            return self._hard_split_spans(text, text_offset)
        sentence_spans = self._sentence_spans(text, text_offset)
        return self._glue_to_length(sentence_spans, self._min_sentence_split)

    def get_max_chunk_size(self, request: TTSRequest, chunk_index: int) -> int:
        if chunk_index >= len(request.chunk_length_schedule):
            return request.chunk_length_schedule[-1]
        return request.chunk_length_schedule[chunk_index]

    def get_min_chunk_size(self, request: TTSRequest, chunk_index: int) -> int:
        if chunk_index >= len(request.min_chunk_length_schedule):
            return request.min_chunk_length_schedule[-1]
        return request.min_chunk_length_schedule[chunk_index]

    def _sentence_spans(self, text: str, text_offset: int) -> list[TextSpan]:
        spans = []
        for part_start, part_end in self._hard_split_ranges(text):
            part = text[part_start:part_end]
            cursor = 0
            for sentence in self._segmenter.segment(part):
                local_start = part.find(sentence, cursor)
                if local_start < 0:
                    raise ValueError(f"pysbd sentence not found in source text: {sentence}")
                local_end = local_start + len(sentence)
                spans.append(
                    TextSpan(
                        text_offset + part_start + local_start,
                        text_offset + part_start + local_end,
                        sentence,
                    )
                )
                cursor = local_end
        return [span for span in spans if span.text.strip()]

    def _hard_split_ranges(self, text: str) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        while True:
            delimiter = text.find("||", start)
            if delimiter < 0:
                ranges.append((start, len(text)))
                return ranges
            ranges.append((start, delimiter))
            start = delimiter + 2

    def _hard_split_spans(self, text: str, text_offset: int) -> list[TextSpan]:
        spans = []
        for part_start, part_end in self._hard_split_ranges(text):
            trimmed = self._trim_span(text, part_start, part_end)
            if trimmed.start == trimmed.end:
                continue
            part_text = text[trimmed.start : trimmed.end]
            if len(part_text) > 300:
                sentence_spans = self._sentence_spans(part_text, text_offset + trimmed.start)
                spans.extend(self._glue_to_length(sentence_spans, self._min_sentence_split))
            else:
                spans.append(TextSpan(text_offset + trimmed.start, text_offset + trimmed.end, part_text.strip()))
        return spans

    def _trim_span(self, text: str, start: int, end: int) -> TextSpan:
        local_start = start
        local_end = end
        while local_start < local_end and text[local_start].isspace():
            local_start += 1
        while local_end > local_start and text[local_end - 1].isspace():
            local_end -= 1
        return TextSpan(local_start, local_end, text[local_start:local_end])

    def _glue_to_length(self, spans: list[TextSpan], min_char_len: int) -> list[TextSpan]:
        glued = []
        for span in spans:
            if glued and len(glued[-1].text.strip()) < min_char_len:
                previous = glued.pop()
                glued.append(TextSpan(previous.start, span.end, f"{previous.text} {span.text}"))
            else:
                glued.append(span)
        return glued
