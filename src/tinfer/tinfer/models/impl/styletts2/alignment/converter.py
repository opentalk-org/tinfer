from __future__ import annotations
import re
from typing import Any

from tinfer.core.request import AlignmentItem, AlignmentType

_PUNCT_ONLY_CHARS = set(".,;:!?¡¿—… \t\n\"\"«»\"\" ")
_MERGE_COST = 1


def _is_punctuation_only(segment_text: str) -> bool:
    return not segment_text or all(c in _PUNCT_ONLY_CHARS for c in segment_text)


def _strip_word_punct(word: str) -> str:
    return re.sub(rf'[{re.escape(";:,.!?¡¿—… \t\n\"\"«»\"\" ")}]+$', '', word)


def _align_segments_to_words_dp(
    segment_texts: list[str],
    original_words: list[str],
) -> list[int]:
    n, m = len(segment_texts), len(original_words)
    is_punct = [not s or _is_punctuation_only(s.strip()) for s in segment_texts]

    dp: list[list[tuple[float, tuple[int, int, str] | None]]] = [
        [(float("inf"), None) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    dp[0][0] = (0.0, None)
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j][0] == float("inf"):
                continue
            c = dp[i][j][0]
            if i < n and is_punct[i]:
                if c < dp[i + 1][j][0]:
                    dp[i + 1][j] = (c, (i, j, "punct"))
            if i < n and j < m and not is_punct[i]:
                if c < dp[i + 1][j + 1][0]:
                    dp[i + 1][j + 1] = (c, (i, j, "one"))
            if i < n and j + 1 < m and not is_punct[i]:
                if c + _MERGE_COST < dp[i + 1][j + 2][0]:
                    dp[i + 1][j + 2] = (c + _MERGE_COST, (i, j, "two"))
    if dp[n][m][1] is None:
        return [1] * min(n, m) if n and m else []
    path: list[int] = []
    i, j = n, m
    while i > 0:
        back = dp[i][j][1]
        if back is None:
            break
        pi, pj, action = back
        if action == "punct":
            path.append(0)
        elif action == "one":
            path.append(1)
        elif action == "two":
            path.append(2)
        i, j = pi, pj
    path.reverse()
    return path


class AlignmentConverter:
    
    @staticmethod
    def phoneme_to_word(
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        phonemized_text: str | None = None,
        word_separator: str = "|",
    ) -> list[AlignmentItem]:

        if not phoneme_alignments:
            return []
        
        if not phonemized_text or word_separator not in phonemized_text:
            raise ValueError(
                "phonemized_text with word separators is required for phoneme-to-word conversion"
            )
        
        return AlignmentConverter._phoneme_to_word_with_separator(
            phoneme_alignments, original_text, phonemized_text, word_separator
        )
    
    @staticmethod
    def _phoneme_to_word_with_separator(
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        phonemized_text: str,
        word_separator: str,
    ) -> list[AlignmentItem]:
        segments = phonemized_text.split(word_separator)
        original_words = original_text.split()
        segment_infos: list[tuple[str, int, int, int, int]] = []
        phoneme_idx = 0

        for segment in segments:
            if not segment:
                continue
            
            segment_phonemes = []
            segment_start = None
            segment_chars = list(segment)
            char_idx = 0
            
            while char_idx < len(segment_chars) and phoneme_idx < len(phoneme_alignments):
                phoneme_item = phoneme_alignments[phoneme_idx]
                
                if phoneme_item.item.strip() == '':
                    phoneme_idx += 1
                    continue
                
                if char_idx < len(segment_chars):
                    expected_char = segment_chars[char_idx]
                    if phoneme_item.item == expected_char:
                        if segment_start is None:
                            segment_start = phoneme_item.start_ms
                        segment_phonemes.append(phoneme_item)
                        char_idx += 1
                        phoneme_idx += 1
                    else:
                        break
                else:
                    break
            if segment_phonemes and segment_start is not None:
                segment_infos.append(
                    (
                        segment.strip(),
                        segment_start,
                        segment_phonemes[-1].end_ms,
                        segment_phonemes[0].char_start,
                        segment_phonemes[-1].char_end,
                    )
                )

            while phoneme_idx < len(phoneme_alignments) and phoneme_alignments[phoneme_idx].item.strip() == "":
                phoneme_idx += 1

        segment_texts = [t[0] for t in segment_infos]
        path = _align_segments_to_words_dp(segment_texts, original_words)
        if len(path) != len(segment_infos):
            word_segment_count = sum(
                1 for s in segment_texts if s and not _is_punctuation_only(s.strip())
            )
            deficit = max(0, len(original_words) - word_segment_count)
            path = []
            words_done = 0
            for s in segment_texts:
                if not s or _is_punctuation_only(s.strip()):
                    path.append(0)
                else:
                    take = 2 if deficit > 0 and words_done + 2 <= len(original_words) else 1
                    path.append(take)
                    words_done += take
                    if take == 2:
                        deficit -= 1

        word_alignments = []
        word_idx = 0
        for k, (seg_text, start_ms, end_ms, char_start, char_end) in enumerate(segment_infos):
            nw = path[k] if k < len(path) else (0 if _is_punctuation_only(seg_text) else 1)
            if nw == 0:
                word_alignments.append(
                    AlignmentItem(
                        item=seg_text if seg_text else "",
                        start_ms=start_ms,
                        end_ms=end_ms,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
            else:
                for _ in range(nw):
                    if word_idx < len(original_words):
                        word_alignments.append(
                            AlignmentItem(
                                item=_strip_word_punct(original_words[word_idx]),
                                start_ms=start_ms,
                                end_ms=end_ms,
                                char_start=char_start,
                                char_end=char_end,
                            )
                        )
                        word_idx += 1
        return word_alignments
    
    @staticmethod
    def word_to_char(
        word_alignments: list[AlignmentItem],
        original_text: str,
    ) -> list[AlignmentItem]:

        char_alignments = []
        char_idx = 0

        for k, word_align in enumerate(word_alignments):
            word_item = getattr(word_align, 'item', getattr(word_align, 'word', ''))
            word_start = word_align.start_ms
            word_duration = word_align.end_ms - word_align.start_ms

            word_chars = list(word_item)
            word_length = len(word_chars)
            if word_length == 0:
                continue

            char_duration = word_duration / word_length

            for i in range(word_length):
                char = word_chars[i]
                char_alignments.append(AlignmentItem(
                    item=char,
                    start_ms=int(word_start + i * char_duration),
                    end_ms=int(word_start + (i + 1) * char_duration),
                    char_start=char_idx,
                    char_end=char_idx + 1,
                ))
                char_idx += 1

            next_idx = k + 1
            if next_idx < len(word_alignments):
                next_item = getattr(
                    word_alignments[next_idx], 'item',
                    getattr(word_alignments[next_idx], 'word', ''),
                )
                if not _is_punctuation_only(next_item.strip()):
                    next_start = word_alignments[next_idx].start_ms
                    char_alignments.append(AlignmentItem(
                        item=' ',
                        start_ms=word_align.end_ms,
                        end_ms=next_start,
                        char_start=char_idx,
                        char_end=char_idx + 1,
                    ))
                    char_idx += 1

        return char_alignments
    
    @staticmethod
    def phoneme_to_char(
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        phonemized_text: str | None = None,
        word_separator: str = "|",
    ) -> list[AlignmentItem]:

        word_alignments = AlignmentConverter.phoneme_to_word(
            phoneme_alignments, original_text, phonemized_text, word_separator
        )
        return AlignmentConverter.word_to_char(word_alignments, original_text)
    
    @staticmethod
    def convert_to(
        alignments: list[AlignmentItem],
        target_type: AlignmentType,
        original_text: str,
        phonemized_text: str | None = None,
        source_type: AlignmentType | None = None,
    ) -> list[AlignmentItem]:
        """
        Convert alignments to target type.
        
        Args:
            alignments: List of alignments (any type)
            target_type: Target alignment type
            original_text: Original input text
            phonemized_text: Phonemized text (for phoneme->word conversion)
            source_type: Source alignment type (if not provided, will be inferred)
            
        Returns:
            List of alignments in target type
        """
        if not alignments:
            return []
        
        if source_type is None:
            raise ValueError("source_type must be provided for conversion")
        
        if source_type == target_type:
            return alignments
        
        if source_type == AlignmentType.PHONEME:
            if target_type == AlignmentType.WORD:
                return AlignmentConverter.phoneme_to_word(
                    alignments, original_text, phonemized_text
                )
            elif target_type == AlignmentType.CHAR:
                return AlignmentConverter.phoneme_to_char(
                    alignments, original_text, phonemized_text
                )
        
        elif source_type == AlignmentType.WORD:
            if target_type == AlignmentType.CHAR:
                return AlignmentConverter.word_to_char(alignments, original_text)
        
        return alignments


