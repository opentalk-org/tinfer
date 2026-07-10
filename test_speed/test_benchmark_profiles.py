from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import tinfer.models.impl.styletts2.model.model as styletts2_model_module
from tinfer.core.request import AlignmentItem
from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params

from test_speed.benchmark_data import RequestMetric, TextInput
from test_speed.benchmark_corpus import build_phoneme_grid
from test_speed.benchmark_inference import measure_result
from test_speed.benchmark_reporting import (
    scatter_coordinates,
    shared_histogram_edges,
)
from test_speed.run_benchmark import disable_speed_correction


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


class PhonemeGridTests(unittest.TestCase):
    def test_grid_has_48_increasing_points_and_maximum_prefix(self) -> None:
        passage = " ".join(f"word{i}" for i in range(80))

        grid = build_phoneme_grid(
            WordTokenModel(),
            passage,
            point_count=48,
            max_tokens=100,
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
        )
        counts = [item.input_phoneme_tokens for item in grid]

        self.assertEqual(len(grid), 48)
        self.assertEqual(counts, sorted(set(counts)))
        self.assertEqual(counts[-1], 493)


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
