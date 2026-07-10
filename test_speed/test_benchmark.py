from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tinfer.core.request import AlignmentItem

from test_speed.benchmark_data import (
    POLISH_INPUTS,
    PhonemeMetric,
    RequestMetric,
    TextInput,
    select_names,
    summarize_phonemes,
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


if __name__ == "__main__":
    unittest.main()
