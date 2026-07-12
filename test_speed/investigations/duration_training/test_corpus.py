import random
import unittest

from test_speed.investigations.duration_training.corpus import (
    DurationSample,
    build_context_pair,
    concatenate_samples,
)


class CorpusTests(unittest.TestCase):
    def test_context_pair_keeps_shared_tokens_and_terminal(self) -> None:
        tokens = [0, *range(10, 30), 4]

        pair = build_context_pair(tokens, short_length=10, terminal_token=4)

        self.assertEqual(pair.long_tokens, tokens)
        self.assertEqual(pair.short_tokens, [*tokens[:9], 4])
        self.assertEqual(pair.shared_length, 9)

    def test_concatenation_scales_partial_segment_duration(self) -> None:
        samples = [
            DurationSample("a", [0, 10, 11, 4], 4.0),
            DurationSample("b", [0, 20, 21, 22, 23, 4], 10.0),
        ]

        combined = concatenate_samples(
            samples,
            maximum_tokens=7,
            terminal_token=4,
            randomizer=random.Random(3),
        )

        self.assertEqual(combined.tokens, [0, 10, 11, 4, 20, 21, 4])
        self.assertAlmostEqual(combined.duration_seconds, 4.0 + 10.0 * 2 / 4)


if __name__ == "__main__":
    unittest.main()
