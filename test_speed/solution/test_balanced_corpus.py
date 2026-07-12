import unittest

from test_speed.solution.balanced_corpus import build_balanced_grid


class WordCountingModel:
    def _text_token_count(self, text, params):
        return len(text.split()) + 1


class BalancedCorpusTests(unittest.TestCase):
    def test_varies_starting_content_across_length_grid(self) -> None:
        passage = "zero one two three four five six seven eight nine"

        rows = build_balanced_grid(
            WordCountingModel(),
            passage,
            point_count=5,
            max_tokens=8,
            language="pl",
            use_training_phonemes=False,
        )

        starts = {row.text.split()[0] for row in rows}
        self.assertEqual(len(rows), 5)
        self.assertGreater(len(starts), 1)
        self.assertTrue(all(row.input_phoneme_tokens <= 8 for row in rows))
        self.assertEqual(len({row.text_id for row in rows}), 5)


if __name__ == "__main__":
    unittest.main()
