import unittest

import numpy as np

from test_speed.investigations.bert_context.metrics import regional_metrics


class RegionalMetricsTest(unittest.TestCase):
    def test_splits_at_one_hundred_and_recovers_slopes(self) -> None:
        tokens = np.array([10, 20, 40, 80, 100, 140, 200, 300])
        rates = np.where(tokens < 100, 5 + 0.1 * tokens, 20 + 0.01 * tokens)

        short, long = regional_metrics(tokens, rates)

        self.assertEqual((short.region, short.count), ("<100", 4))
        self.assertEqual((long.region, long.count), (">=100", 4))
        self.assertAlmostEqual(short.slope, 0.1)
        self.assertAlmostEqual(long.slope, 0.01)


if __name__ == "__main__":
    unittest.main()
