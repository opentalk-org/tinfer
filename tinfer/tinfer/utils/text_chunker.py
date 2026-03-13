from __future__ import annotations
from typing import Literal, Optional, List
import re
from tinfer.core.request import TTSRequest
from time import monotonic

class TextChunker:
    def __init__(self) -> None:
        pass

    def get_chunks(self, request: TTSRequest, single_chunk: bool = False):
        now = monotonic()

        pending_text = request.get_pending_text()

        should_trigger = request.should_trigger_now(now)
        if not should_trigger:
            return []

        current_chunk_index = request.chunker_state.get("chunk_index", 0)


        # TODO: better handling for chunking schedule
        chunks = self.split_text_if_needed(pending_text, request, current_chunk_index)

        if single_chunk:
            is_final = len(chunks) == 1
            chunk = chunks[0]
            text_span_start = request.text_committed_pos
            text_span_end = request.text_committed_pos + len(chunk)
            request.chunker_state["chunk_index"] = current_chunk_index + 1
            return [(chunk, current_chunk_index, is_final, (text_span_start, text_span_end))]

        results = []
        offset = 0
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            text_span_start = request.text_committed_pos + offset
            text_span_end = request.text_committed_pos + offset + len(chunk)
            results.append((chunk, current_chunk_index, is_final, (text_span_start, text_span_end)))
            offset += len(chunk)
            current_chunk_index += 1

        
        request.chunker_state["chunk_index"] = current_chunk_index
        return results

    def split_text_if_needed(
        self, text: str, request: TTSRequest, chunk_index: int
    ) -> List[str]:
        max_size = self.get_max_chunk_size(request, chunk_index)
        if len(text) <= max_size:
            return [text]
        
        separators = ["\n\n", "\n", r"(?<=[.!?]) +", r"(?<=[,;]) +", " "]
        return self._split_text_recursive(text, request, chunk_index, separators, 0)

    def get_max_chunk_size(self, request: TTSRequest, chunk_index: int) -> int:
        """Get max chunk size for current chunk index."""
        if chunk_index >= len(request.chunk_length_schedule):
            return request.chunk_length_schedule[-1]
        return request.chunk_length_schedule[chunk_index]

    def get_min_chunk_size(self, request: TTSRequest, chunk_index: int) -> int:
        """Get min chunk size for current chunk index."""
        if chunk_index >= len(request.min_chunk_length_schedule):
            return request.min_chunk_length_schedule[-1]
        return request.min_chunk_length_schedule[chunk_index]

    def _split_text_recursive(
        self,
        text: str,
        request: TTSRequest,
        chunk_index: int = 0,
        separators: Optional[List[str]] = None,
        recursion_depth: int = 0,
    ) -> List[str]:
        if recursion_depth > 100:
            chunks = []
            if chunk_index >= len(request.chunk_length_schedule):
                max_size = request.chunk_length_schedule[-1]
            else:
                max_size = request.chunk_length_schedule[chunk_index]
            for i in range(0, len(text), max_size):
                chunk = text[i : i + max_size]
                if chunk:
                    chunks.append(chunk)
            return chunks
        
        if chunk_index >= len(request.chunk_length_schedule):
            max_size = request.chunk_length_schedule[-1]
            min_size = request.min_chunk_length_schedule[-1]
        else:
            max_size = request.chunk_length_schedule[chunk_index]
            min_size = request.min_chunk_length_schedule[chunk_index]
        
        if len(text) <= max_size:
            return [text]

        if not separators:
            chunks = []
            for i in range(0, len(text), max_size):
                chunk = text[i : i + max_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        current_sep = separators[0]
        remaining_seps = separators[1:]
        
        chunks = []

        if current_sep != " ":
            if not (current_sep.startswith('(') and current_sep.endswith(')')):
                split_pattern = f"({current_sep})"
            else:
                split_pattern = current_sep
            parts = re.split(split_pattern, text)
        else:
            parts = text.split(" ")

        current_chunk = ""

        i = 0
        while i < len(parts):
            part = parts[i]
            sep_to_add = ""

            if current_sep != " " and i + 1 < len(parts):
                sep_to_add = parts[i+1]
            merged = part + sep_to_add

            step = 2 if sep_to_add else 1

            current_chunk_len = len(current_chunk)
            space_needed = 1 if current_chunk and current_sep == " " else 0
            potential_len = current_chunk_len + len(merged) + space_needed

            if len(merged) > max_size:
                if current_chunk and len(current_chunk.strip()) >= min_size:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = ""
                elif current_chunk:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = ""
                sub_chunks = self._split_text_recursive(merged, request, chunk_index, remaining_seps, recursion_depth + 1)
                chunks.extend(sub_chunks)
            elif potential_len > max_size:
                if current_chunk and len(current_chunk.strip()) >= min_size:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = merged
                elif current_chunk:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = merged
                else:
                    current_chunk = merged
            else:
                if current_chunk and current_sep == " ":
                    current_chunk = f"{current_chunk} {merged}"
                else:
                    current_chunk = f"{current_chunk}{merged}"
            i += step

        if current_chunk:
            chunks.append(current_chunk.rstrip())

        result = [c.strip() for c in chunks if c.strip()]
        
        final_chunks = []
        next_chunk_index = chunk_index + 1
        for chunk in result:
            if len(chunk) > max_size:
                if remaining_seps:
                    final_chunks.extend(self._split_text_recursive(chunk, request, next_chunk_index, remaining_seps, recursion_depth + 1))
                else:
                    chunks_fixed = []
                    for i in range(0, len(chunk), max_size):
                        chunk_part = chunk[i : i + max_size]
                        if chunk_part:
                            chunks_fixed.append(chunk_part)
                    final_chunks.extend(chunks_fixed)
            elif len(chunk) < min_size and final_chunks:
                final_chunks[-1] = final_chunks[-1] + chunk
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def find_sentence_boundary(self, text: str, start_pos: int) -> int:
        if start_pos >= len(text):
            return len(text)
        
        pattern = r'[.!?]\s+'
        match = re.search(pattern, text[start_pos:])
        if match:
            return start_pos + match.start()
        return len(text)

    def find_punctuation_boundary(self, text: str, start_pos: int) -> int:
        if start_pos >= len(text):
            return len(text)
        
        pattern = r'[.,;:!?]\s*'
        match = re.search(pattern, text[start_pos:])
        if match:
            return start_pos + match.start()
        return len(text)

