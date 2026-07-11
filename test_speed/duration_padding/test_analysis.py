import unittest

import numpy as np

from test_speed.duration_padding.analysis import compare_real_tokens


class CompareRealTokensTests(unittest.TestCase):
    def test_compares_only_real_token_prefix(self) -> None:
        baseline = np.array([[1.0, 2.0], [3.0, 4.0]])
        padded = np.array([[1.0, 2.5], [3.0, 4.0], [99.0, 99.0]])

        comparison = compare_real_tokens(baseline, padded, real_length=2)

        self.assertEqual(comparison.max_absolute_difference, 0.5)
        self.assertEqual(comparison.mean_absolute_difference, 0.125)
        np.testing.assert_array_equal(
            comparison.per_token_maximum,
            np.array([0.5, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
