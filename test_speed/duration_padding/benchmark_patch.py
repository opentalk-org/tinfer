from types import MethodType
from typing import Any

import torch
from torch import nn


def packed_lstm_forward(
    lstm: nn.LSTM,
    inputs: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    lengths = inputs.abs().sum(dim=-1).ne(0).sum(dim=-1).cpu()
    packed = nn.utils.rnn.pack_padded_sequence(
        inputs,
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, state = lstm.unpacked_forward(packed, hidden)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=inputs.shape[1],
    )
    return output, state


def install_packed_lstm_forward(model: Any) -> None:
    lstm = model.predictor.lstm
    lstm.unpacked_forward = lstm.forward
    lstm.forward = MethodType(packed_lstm_forward, lstm)
