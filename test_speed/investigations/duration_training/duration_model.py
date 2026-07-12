from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor


def make_token_batch(sequences: list[list[int]], device: str) -> TokenBatch:
    lengths = torch.tensor([len(tokens) for tokens in sequences], device=device)
    maximum = int(lengths.max())
    padded = [tokens + [0] * (maximum - len(tokens)) for tokens in sequences]
    tensor = torch.tensor(padded, dtype=torch.long, device=device)
    positions = torch.arange(maximum, device=device).unsqueeze(0)
    return TokenBatch(tensor, lengths, positions >= lengths.unsqueeze(1))


def duration_forward(
    bert_encoder: nn.Module,
    predictor: nn.Module,
    bert_output: torch.Tensor,
    style: torch.Tensor,
    batch: TokenBatch,
) -> torch.Tensor:
    encoded = bert_encoder(bert_output).transpose(-1, -2)
    duration_encoded = predictor.text_encoder(
        encoded,
        style[:, 128:],
        batch.lengths,
        batch.mask,
    )
    packed = nn.utils.rnn.pack_padded_sequence(
        duration_encoded,
        batch.lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed, _ = predictor.lstm(packed)
    recurrent, _ = nn.utils.rnn.pad_packed_sequence(
        packed,
        batch_first=True,
        total_length=batch.tokens.shape[1],
    )
    logits = predictor.duration_proj(recurrent)
    return torch.sigmoid(logits).sum(dim=-1)


def total_duration_loss(
    durations: torch.Tensor,
    lengths: torch.Tensor,
    target_frames: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(durations.shape[1], device=durations.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    predicted = (durations * valid).sum(dim=1)
    return F.smooth_l1_loss(torch.log1p(predicted), torch.log1p(target_frames))


def centered_shape_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for row, target, length in zip(student, teacher, lengths.tolist()):
        predicted_log = torch.log(row[1 : length - 1])
        target_log = torch.log(target[1 : length - 1])
        losses.append(
            F.smooth_l1_loss(
                predicted_log - predicted_log.mean(),
                target_log - target_log.mean(),
            )
        )
    return torch.stack(losses).mean()


def consistency_loss(
    short: torch.Tensor,
    full: torch.Tensor,
    shared_lengths: torch.Tensor,
    boundary: int,
) -> torch.Tensor:
    losses = []
    for short_row, full_row, shared in zip(short, full, shared_lengths.tolist()):
        end = shared - boundary
        losses.append(
            F.smooth_l1_loss(
                torch.log(short_row[1:end]),
                torch.log(full_row[1:end].detach()),
            )
        )
    return torch.stack(losses).mean()
