from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import soundfile as sf

import tinfer.models.impl.styletts2.model.model as styletts2_model_module
from tinfer.core.request import AlignmentItem
from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer

import test_speed.benchmark_corpus as corpus
from test_speed.benchmark_data import ReferenceDuration, RequestMetric, TextInput
from test_speed.benchmark_corpus import build_phoneme_grid
from test_speed.benchmark_inference import measure_reference_durations, measure_result
from test_speed.benchmark_reporting import (
    reference_all_run_coordinates,
    reference_mean_voice_coordinates,
    scatter_coordinates,
    shared_histogram_edges,
)
from test_speed.run_benchmark import disable_speed_correction


class TrainingPhonemizerTests(unittest.TestCase):
    def test_matches_backend_training_phonemes(self) -> None:
        phonemize = getattr(corpus, "phonemize_training_text", None)
        self.assertIsNotNone(phonemize)
        assert phonemize is not None

        result = phonemize("W tamtym momencie, kiedy wiedząc.", "pl")

        self.assertEqual(result, "f tˈamtɨm mɔmˈɛɲtɕɛ, kʲˈɛdɨ vʲˈɛdʑɔnts.")

    def test_checkpoint_symbols_replace_phonemizer_indices(self) -> None:
        configure = getattr(corpus, "configure_checkpoint_symbols", None)
        self.assertIsNotNone(configure)
        assert configure is not None
        phonemizer = StyleTTS2Phonemizer(language="pl")
        model = SimpleNamespace(
            _config={"symbols": ["$", "x", "ʲ"]},
            _phonemizers={"pl": phonemizer},
        )

        configure(model, "pl")

        self.assertEqual(phonemizer.word_index_dictionary, {"$": 0, "x": 1, "ʲ": 2})


class WordTokenModel:
    def _text_token_count(
        self,
        text: str,
        params: StyleTTS2Params,
    ) -> int:
        return len(text.split()) * 2 + 1


class GappedWordTokenModel:
    def _text_token_count(
        self,
        text: str,
        params: StyleTTS2Params,
    ) -> int:
        word_count = len(text.split())
        token_count = 10 if word_count == 1 else 23 + (word_count - 2) * 10
        return token_count + 1


class LanguageRecordingTokenModel:
    def __init__(self) -> None:
        self.languages: list[str | None] = []

    def _text_token_count(
        self,
        text: str,
        params: StyleTTS2Params,
    ) -> int:
        self.languages.append(params.language)
        return len(text.split()) * 2 + 1


class PhonemeGridTests(unittest.TestCase):
    def test_grid_has_48_increasing_points_and_maximum_prefix(self) -> None:
        passage = " ".join(f"word{i}" for i in range(80))

        grid = build_phoneme_grid(
            WordTokenModel(),
            passage,
            point_count=48,
            max_tokens=100,
            language="pl",
            use_training_phonemes=False,
        )
        counts = [item.input_phoneme_tokens for item in grid]

        self.assertEqual(len(grid), 48)
        self.assertEqual(counts, sorted(set(counts)))
        self.assertEqual(counts[-1], 100)
        self.assertEqual(len(grid[-1].text.split()), 50)

    def test_grid_stays_unique_across_large_early_token_gap(self) -> None:
        passage = " ".join(f"word{i}" for i in range(60))

        grid = build_phoneme_grid(
            GappedWordTokenModel(),
            passage,
            point_count=48,
            max_tokens=500,
            language="pl",
            use_training_phonemes=False,
        )
        counts = [item.input_phoneme_tokens for item in grid]

        self.assertEqual(len(grid), 48)
        self.assertEqual(counts, sorted(set(counts)))
        self.assertEqual(counts[-1], 493)

    def test_grid_counts_with_target_language(self) -> None:
        model = LanguageRecordingTokenModel()
        passage = " ".join(f"word{i}" for i in range(80))

        build_phoneme_grid(
            model,
            passage,
            point_count=48,
            max_tokens=100,
            language="en-us",
            use_training_phonemes=False,
        )

        self.assertEqual(set(model.languages), {"en-us"})


class PhonemeMetricTests(unittest.TestCase):
    def test_result_preserves_input_phoneme_tokens(self) -> None:
        result = SimpleNamespace(
            metadata={
                "word_alignments": [AlignmentItem("a", 0, 1, 0, 25)]
            }
        )
        text_input = TextInput("phonemes_007", "tekst", 7)

        request, _ = measure_result(
            result,
            "voice",
            text_input,
            Path("a.wav"),
        )

        self.assertEqual(request.input_phoneme_tokens, 7)

    def test_scatter_uses_input_phoneme_tokens(self) -> None:
        metrics = [
            RequestMetric("v", "t", "x", 1, 7, 1, 0.05, 20.0, "a.wav")
        ]

        self.assertEqual(scatter_coordinates(metrics), ([7], [20.0]))


class ReferenceDurationPlotTests(unittest.TestCase):
    def test_reference_duration_is_measured_from_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.wav"
            sf.write(path, np.zeros(12_000, dtype=np.float32), 24_000)

            references = measure_reference_durations([path])

        self.assertEqual(references[0].voice_id, "voice")
        self.assertAlmostEqual(references[0].duration_seconds, 0.5)

    def test_reference_scatter_has_all_runs_and_voice_means(self) -> None:
        references = [ReferenceDuration("a", 2.0), ReferenceDuration("b", 4.0)]
        metrics = [
            RequestMetric("a", "x", "x", 1, 7, 1, 0.05, 20.0, "a.wav"),
            RequestMetric("a", "y", "y", 1, 9, 1, 0.025, 40.0, "b.wav"),
            RequestMetric("b", "x", "x", 1, 7, 1, 0.05, 30.0, "c.wav"),
        ]

        self.assertEqual(
            reference_all_run_coordinates(metrics, references),
            ([2.0, 2.0, 4.0], [20.0, 40.0, 30.0]),
        )
        self.assertEqual(
            reference_mean_voice_coordinates(metrics, references),
            ([2.0, 4.0], [30.0, 30.0]),
        )


class SpeedCorrectionTests(unittest.TestCase):
    def test_speed_correction_is_disabled_for_benchmark(self) -> None:
        original = styletts2_model_module.baseline_speed_corrected_for_request
        try:
            disable_speed_correction()
            corrected = (
                styletts2_model_module.baseline_speed_corrected_for_request(
                    1.0,
                    300,
                )
            )
            self.assertEqual(corrected, 1.0)
        finally:
            styletts2_model_module.baseline_speed_corrected_for_request = original


class HistogramBinTests(unittest.TestCase):
    def test_shared_edges_use_quarter_phoneme_per_second_bins(self) -> None:
        edges = shared_histogram_edges(
            [9.1],
            [10.1],
            bin_width=0.25,
        )

        np.testing.assert_array_equal(
            edges,
            np.arange(9.0, 10.5, 0.25),
        )


if __name__ == "__main__":
    unittest.main()
