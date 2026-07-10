from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import numpy as np

from tinfer.core.request import AlignmentItem
from tinfer.models.impl.styletts2.alignment.alignment import StyleTTS2AlignmentParser

from test_speed.benchmark_data import (
    POLISH_INPUTS,
    PhonemeMetric,
    RequestMetric,
    TextInput,
    select_names,
    summarize_phonemes,
)
from test_speed.benchmark_inference import (
    archive_wav_names,
    extract_selected,
    measure_result,
)


class DataTests(unittest.TestCase):
    def test_selection_is_seeded_and_sorted(self) -> None:
        names = [f"v{i}" for i in range(30)]
        selected = select_names(names, 20, 7)

        self.assertEqual(selected, select_names(names, 20, 7))
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(len(selected), 20)

    def test_selection_rejects_insufficient_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Need 20 inputs, found 1"):
            select_names(["only"], 20, 7)

    def test_summary_groups_each_phoneme(self) -> None:
        rows = summarize_phonemes(
            [
                PhonemeMetric("a", 0.025, "v", "t"),
                PhonemeMetric("a", 0.075, "v", "t"),
            ]
        )

        self.assertEqual(rows[0].count, 2)
        self.assertAlmostEqual(rows[0].average_seconds, 0.05)
        self.assertAlmostEqual(rows[0].minimum_seconds, 0.025)
        self.assertAlmostEqual(rows[0].maximum_seconds, 0.075)

    def test_corpus_spans_two_to_three_hundred_characters(self) -> None:
        lengths = [len(item.text) for item in POLISH_INPUTS]

        self.assertEqual(len(POLISH_INPUTS), 16)
        self.assertEqual(lengths[0], 2)
        self.assertEqual(lengths[-1], 300)
        self.assertEqual(lengths, sorted(lengths))


class InferenceTests(unittest.TestCase):
    def test_phoneme_alignment_excludes_beginning_of_sequence_duration(self) -> None:
        predictor_alignment = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )

        alignments = StyleTTS2AlignmentParser().parse_from_pred_aln_trg(
            predictor_alignment,
            [1],
            "a",
            "a",
        )

        self.assertEqual(len(alignments), 1)
        self.assertEqual(alignments[0].item, "a")
        self.assertEqual(alignments[0].end_ms - alignments[0].start_ms, 50)

    def test_measure_result_uses_predictor_alignment(self) -> None:
        metadata = {"word_alignments": [AlignmentItem("a", 0, 1, 0, 25)]}
        result = SimpleNamespace(metadata=metadata)

        request, phonemes = measure_result(
            result,
            "v",
            TextInput("short", "No"),
            Path("a.wav"),
        )

        self.assertEqual(request.phoneme_count, 1)
        self.assertAlmostEqual(request.phonemes_per_second, 40.0)
        self.assertAlmostEqual(phonemes[0].duration_seconds, 0.025)

    def test_merged_model_windows_are_rejected(self) -> None:
        result = SimpleNamespace(
            metadata={"window_count": 2, "word_alignments": []}
        )

        with self.assertRaisesRegex(RuntimeError, "chunked"):
            measure_result(
                result,
                "v",
                TextInput("short", "No"),
                Path("a.wav"),
            )

    def test_archive_filters_metadata_and_extracts_selected_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "voices.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("voice.wav", b"RIFFdata")
                bundle.writestr("__MACOSX/._voice.wav", b"metadata")
                bundle.writestr("notes.txt", b"ignore")

            self.assertEqual(archive_wav_names(archive), ["voice.wav"])
            extracted = extract_selected(archive, ["voice.wav"], root / "out")

            self.assertEqual(extracted, [root / "out/voice.wav"])
            self.assertEqual(extracted[0].read_bytes(), b"RIFFdata")


if __name__ == "__main__":
    unittest.main()
