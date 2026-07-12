import unittest

import torch
from torch import nn

from test_speed.investigations.padding.benchmark_patch import install_packed_lstm_forward


class PredictorStub:
    def __init__(self) -> None:
        torch.manual_seed(7)
        self.lstm = nn.LSTM(3, 2, batch_first=True, bidirectional=True)


class ModelStub:
    def __init__(self) -> None:
        self.predictor = PredictorStub()


class PackedLstmPatchTests(unittest.TestCase):
    def test_real_outputs_do_not_depend_on_zero_right_padding(self) -> None:
        model = ModelStub()
        install_packed_lstm_forward(model)
        real = torch.randn(1, 4, 3)
        short, _ = model.predictor.lstm(real)
        long, _ = model.predictor.lstm(
            torch.cat([real, torch.zeros(1, 5, 3)], dim=1)
        )

        torch.testing.assert_close(short, long[:, :4], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(long[:, 4:], torch.zeros_like(long[:, 4:]))


if __name__ == "__main__":
    unittest.main()
