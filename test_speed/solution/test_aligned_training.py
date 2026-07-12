import unittest

import torch

from test_speed.solution.aligned_training import aligned_duration_loss
from test_speed.solution.rate_regularized_training import (
    phonemes_per_second,
    rate_variance_loss,
)


class AlignedTrainingTests(unittest.TestCase):
    def test_loss_uses_real_tokens_and_excludes_bos_padding(self) -> None:
        predictions = torch.tensor(
            [[99.0, 2.0, 3.0, 100.0], [88.0, 4.0, 100.0, 100.0]]
        )
        targets = [torch.tensor([2.0, 3.0]), torch.tensor([4.0])]

        loss = aligned_duration_loss(predictions, targets)

        self.assertEqual(float(loss), 0.0)

    def test_rate_variance_is_zero_for_equal_spoken_means(self) -> None:
        predictions = torch.tensor([[99.0, 2.0, 4.0], [88.0, 3.0, 3.0]])
        spoken_masks = [torch.tensor([True, True]), torch.tensor([True, True])]

        loss = rate_variance_loss(predictions, spoken_masks)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_rate_uses_training_frame_timebase(self) -> None:
        prediction = torch.tensor([99.0, 2.0, 2.0])
        spoken = torch.tensor([True, True])

        rate = phonemes_per_second(prediction, spoken)

        self.assertEqual(rate, 20.0)

if __name__ == "__main__":
    unittest.main()
