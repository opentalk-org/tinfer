from __future__ import annotations

import unittest

import espeak_align
import numpy as np

from tinfer.core.request import AlignmentItem
from tinfer.core.request import TTSRequest
from tinfer.core.stream import TTSStream
from tinfer.models.impl.styletts2.alignment.converter import AlignmentConverter
from tinfer.models.impl.styletts2.model.model_utils import (
    _trim_leading_silence_and_shift_alignments,
)
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer
from tinfer.utils.text_chunker import TextChunker


class AlignmentSpanTests(unittest.TestCase):
    def test_rust_align_with_spans_returns_preprocessed_token_offsets(self) -> None:
        engine = espeak_align.Engine("pl", tie=True, espeak_workers=1)

        items = engine.align_with_spans("Wyślij https:a—b.pl", ";:,.!?¡¿—…\"«»\"\"", threads=1)

        self.assertTrue(items)
        for item in items:
            token = item["token"]
            self.assertEqual("Wyślij https:a—b.pl"[item["start"] : item["end"]], token)

        tokens = [item["token"] for item in items]
        self.assertIn("https", tokens)
        self.assertIn(":", tokens)
        self.assertIn("—", tokens)

    def test_rust_align_preserves_punctuation_order(self) -> None:
        engine = espeak_align.Engine("pl", tie=True, espeak_workers=1)
        text = '"Cześć" — powiedziała: "To działa… prawda?"'

        words, _phonemes = engine.align(text, ";:,.!?¡¿—…\"«»\"\"", threads=1)

        self.assertEqual("".join(words), text)

    def test_phonemizer_maps_preprocessed_spans_back_to_original_text(self) -> None:
        phonemizer = StyleTTS2Phonemizer(language="pl", espeak_workers=1)
        original = "Wyślij https://a-b.pl/x"

        mapped = phonemizer.align_text_with_original_spans(original)

        self.assertTrue(mapped)
        self.assertEqual("".join(item["original_text"] for item in mapped), original)
        for item in mapped:
            self.assertEqual(
                original[item["original_start"] : item["original_end"]],
                item["original_text"],
            )

    def test_phonemizer_preserves_leading_space_in_original_spans(self) -> None:
        phonemizer = StyleTTS2Phonemizer(language="pl", espeak_workers=1)
        original = " 8:05; patrz https://example.org/a-b."

        mapped = phonemizer.align_text_with_original_spans(original)

        self.assertEqual("".join(item["original_text"] for item in mapped), original)
        self.assertEqual(mapped[0]["original_start"], 0)

    def test_phonemizer_maps_curly_quotes_without_panicking(self) -> None:
        phonemizer = StyleTTS2Phonemizer(language="pl", espeak_workers=1)
        original = "„Cześć” — powiedziała: «To działa… prawda?»"

        mapped = phonemizer.align_text_with_original_spans(original)

        self.assertEqual("".join(item["original_text"] for item in mapped), original)

    def test_converter_reconstructs_original_chars_from_mapped_spans(self) -> None:
        phonemizer = StyleTTS2Phonemizer(language="pl", espeak_workers=1)
        original = "Wyślij https://a-b.pl/x"
        mapped = phonemizer.align_text_with_original_spans(original)
        phoneme_count = sum(len(item["phonemes"]) for item in mapped)
        phoneme_alignments = [
            AlignmentItem(
                item="x",
                start_ms=i * 10,
                end_ms=(i + 1) * 10,
                char_start=i,
                char_end=i + 1,
            )
            for i in range(phoneme_count)
        ]

        char_alignments = AlignmentConverter.phoneme_to_char_mapped(
            phoneme_alignments,
            original,
            mapped,
        )

        self.assertEqual("".join(item.item for item in char_alignments), original)
        for item in char_alignments:
            self.assertEqual(original[item.char_start : item.char_end], item.item)

    def test_chunker_spans_match_returned_chunk_text(self) -> None:
        text = "Dr inż. Kowalski pisał do test@example.com o godz. 8:05; patrz https://example.org/a-b."
        request = TTSRequest(
            request_id="test",
            model_id="styletts2",
            voice_id="voice",
            chunk_length_schedule=[50],
            timeout_trigger_ms=0,
        )
        request.append_text(text)

        chunks = TextChunker().get_chunks(request)

        self.assertGreater(len(chunks), 1)
        for chunk, _chunk_index, _is_final, span in chunks:
            self.assertEqual(text[span[0] : span[1]], chunk)

    def test_chunker_preserves_sentence_boundary_whitespace(self) -> None:
        text = (
            "In case you don't know, you can run /side or just use the plus button "
            "on the top right to open side conversations. This is great for context "
            'management, going down rabbit holes, side-questions like "wait why did '
            'you do it that way", etc'
        )
        request = TTSRequest(
            request_id="test",
            model_id="libri",
            voice_id="libri_m1",
            chunk_length_schedule=[120, 160, 250, 290],
            timeout_trigger_ms=0,
        )
        request.append_text(text)

        chunks = TextChunker().get_chunks(request)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0][0].endswith("conversations. "))
        self.assertTrue(chunks[1][0].startswith("This is great"))
        self.assertEqual("".join(chunk[0] for chunk in chunks), text)
        for chunk, _chunk_index, _is_final, span in chunks:
            self.assertEqual(text[span[0] : span[1]], chunk)

    def test_get_audio_returns_only_currently_queued_chunks(self) -> None:
        class Engine:
            def signal_input(self) -> None:
                raise AssertionError("get_audio must not trigger scheduling")

        request = TTSRequest(
            request_id="test",
            model_id="styletts2",
            voice_id="voice",
        )
        request.pending_chunks = 1
        stream = TTSStream(request, Engine())

        self.assertEqual(stream.get_audio(), [])

    def test_leading_silence_trim_shifts_alignment_times(self) -> None:
        sample_rate = 1000
        audio = np.concatenate([np.zeros(300, dtype=np.float32), np.ones(700, dtype=np.float32) * 0.2])
        alignments = [
            AlignmentItem(item="a", char_start=0, char_end=1, start_ms=0, end_ms=200),
            AlignmentItem(item="b", char_start=1, char_end=2, start_ms=300, end_ms=500),
        ]

        trimmed_audio, shifted = _trim_leading_silence_and_shift_alignments(audio, sample_rate, alignments)

        self.assertLess(len(trimmed_audio), len(audio))
        self.assertEqual(shifted[0].start_ms, 0)
        self.assertEqual(shifted[0].end_ms, 0)
        self.assertEqual(shifted[1].start_ms, 0)
        self.assertEqual(shifted[1].end_ms, 200)


if __name__ == "__main__":
    unittest.main()
