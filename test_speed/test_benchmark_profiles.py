import unittest

import numpy as np

import tinfer.models.impl.styletts2.model.model as styletts2_model_module

from test_speed.benchmark_reporting import shared_histogram_edges
from test_speed.run_benchmark import disable_speed_correction


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
