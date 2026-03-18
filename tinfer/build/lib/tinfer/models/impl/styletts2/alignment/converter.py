from __future__ import annotations

from tinfer.core.request import AlignmentItem, AlignmentType

_PUNCT_ONLY_CHARS = set(".,;:!?¡¿—… \t\n\"\"«»\"\" ")


def _is_punctuation_only(segment_text: str) -> bool:
    return not segment_text or all(c in _PUNCT_ONLY_CHARS for c in segment_text)


class AlignmentConverter:
    @staticmethod
    def phoneme_to_word(
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        words: list[str],
        phonemes_list: list[str],
    ) -> list[AlignmentItem]:
        if not phoneme_alignments or not words or not phonemes_list:
            return []
        if len(words) != len(phonemes_list):
            return []


        n_phonemes_per_word = [len(p) for p in phonemes_list]
        if sum(n_phonemes_per_word) != len(phoneme_alignments):
            return []

        word_alignments: list[AlignmentItem] = []
        idx = 0
        for i, n in enumerate(n_phonemes_per_word):
            segment = phoneme_alignments[idx : idx + n]
            idx += n
            if not segment:
                continue
            word_alignments.append(
                AlignmentItem(
                    item=words[i],
                    start_ms=segment[0].start_ms,
                    end_ms=segment[-1].end_ms,
                    char_start=segment[0].char_start,
                    char_end=segment[-1].char_end,
                )
            )
        return word_alignments

    @staticmethod
    def word_to_char(
        word_alignments: list[AlignmentItem],
        original_text: str,
    ) -> list[AlignmentItem]:
        char_alignments: list[AlignmentItem] = []
        char_idx = 0

        for k, word_align in enumerate(word_alignments):
            word_item = getattr(word_align, "item", getattr(word_align, "word", ""))
            word_start = word_align.start_ms
            word_duration = word_align.end_ms - word_align.start_ms
            word_chars = list(word_item)
            word_length = len(word_chars)
            if word_length == 0:
                continue

            char_duration = word_duration / word_length
            for i in range(word_length):
                char_alignments.append(
                    AlignmentItem(
                        item=word_chars[i],
                        start_ms=int(word_start + i * char_duration),
                        end_ms=int(word_start + (i + 1) * char_duration),
                        char_start=char_idx,
                        char_end=char_idx + 1,
                    )
                )
                char_idx += 1

            if k + 1 < len(word_alignments):
                next_item = getattr(
                    word_alignments[k + 1], "item", getattr(word_alignments[k + 1], "word", "")
                )
                if not _is_punctuation_only(next_item.strip()):
                    next_start = word_alignments[k + 1].start_ms
                    char_alignments.append(
                        AlignmentItem(
                            item=" ",
                            start_ms=word_align.end_ms,
                            end_ms=next_start,
                            char_start=char_idx,
                            char_end=char_idx + 1,
                        )
                    )
                    char_idx += 1

        return char_alignments

    @staticmethod
    def phoneme_to_char(
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        words: list[str],
        phonemes_list: list[str],
    ) -> list[AlignmentItem]:
        word_alignments = AlignmentConverter.phoneme_to_word(
            phoneme_alignments, original_text, words, phonemes_list
        )
        return AlignmentConverter.word_to_char(word_alignments, original_text)

    @staticmethod
    def convert_to(
        alignments: list[AlignmentItem],
        target_type: AlignmentType,
        original_text: str,
        words: list[str] | None = None,
        phonemes_list: list[str] | None = None,
        source_type: AlignmentType | None = None,
    ) -> list[AlignmentItem]:
        if not alignments or source_type is None:
            return []
        if source_type == target_type:
            return alignments

        if source_type == AlignmentType.PHONEME:
            if words is None or phonemes_list is None:
                return []
            if target_type == AlignmentType.WORD:
                return AlignmentConverter.phoneme_to_word(
                    alignments, original_text, words, phonemes_list
                )
            if target_type == AlignmentType.CHAR:
                return AlignmentConverter.phoneme_to_char(
                    alignments, original_text, words, phonemes_list
                )

        if source_type == AlignmentType.WORD and target_type == AlignmentType.CHAR:
            return AlignmentConverter.word_to_char(alignments, original_text)

        return []
