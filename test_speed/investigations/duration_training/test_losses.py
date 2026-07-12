import unittest

import torch

from test_speed.investigations.duration_training.duration_model import (
    centered_shape_loss,
    consistency_loss,
    total_duration_loss,
)


class DurationLossTests(unittest.TestCase):
    def test_total_loss_ignores_padding(self) -> None:
        durations = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
        lengths = torch.tensor([3])

        loss = total_duration_loss(durations, lengths, torch.tensor([6.0]))

        self.assertEqual(float(loss), 0.0)

    def test_shape_loss_ignores_uniform_rate_scaling(self) -> None:
        student = torch.tensor([[2.0, 4.0, 8.0, 1.0]])
        teacher = student * 3

        loss = centered_shape_loss(student, teacher, torch.tensor([4]))

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_consistency_uses_only_shared_interior(self) -> None:
        short = torch.tensor([[99.0, 2.0, 4.0, 8.0, 99.0]])
        full = torch.tensor([[77.0, 2.0, 4.0, 80.0, 77.0, 10.0]])

        loss = consistency_loss(short, full, torch.tensor([4]), boundary=1)

        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
