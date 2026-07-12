from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tinfer.models.impl.styletts2.model.modules.blocks import AdaLayerNorm


@dataclass(frozen=True)
class TensorComparison:
    max_absolute_difference: float
    mean_absolute_difference: float
    per_token_maximum: np.ndarray


@dataclass(frozen=True)
class DurationTrace:
    stages: dict[str, torch.Tensor]


def compare_real_tokens(
    baseline: np.ndarray,
    padded: np.ndarray,
    real_length: int,
) -> TensorComparison:
    difference = np.abs(baseline[:real_length] - padded[:real_length])
    feature_axes = tuple(range(1, difference.ndim))
    per_token = difference.max(axis=feature_axes) if feature_axes else difference
    return TensorComparison(
        float(difference.max()),
        float(difference.mean()),
        np.asarray(per_token),
    )


def _run_conditioned_encoder(
    encoder: nn.Module,
    encoded: torch.Tensor,
    style: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masks = mask.to(lengths.device)
    x = encoded.permute(2, 0, 1)
    expanded_style = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, expanded_style], dim=-1)
    x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1).transpose(-1, -2)
    stages = {}
    lstm_index = 0
    for block in encoder.lstms:
        if isinstance(block, AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, expanded_style.permute(1, -1, 0)], dim=1)
            x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            continue
        packed = nn.utils.rnn.pack_padded_sequence(
            x.transpose(-1, -2),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        block.flatten_parameters()
        packed_output, _ = block(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=mask.shape[-1],
        )
        output = F.dropout(output, p=encoder.dropout, training=encoder.training)
        lstm_index += 1
        stages[f"predictor.text_encoder.bilstm_{lstm_index}"] = output
        x = output.transpose(-1, -2)
    return x.transpose(-1, -2), stages


def _run_final_lstm(
    predictor: nn.Module,
    encoded: torch.Tensor,
    lengths: torch.Tensor,
    padded_length: int,
    packed: bool,
) -> torch.Tensor:
    predictor.lstm.flatten_parameters()
    if not packed:
        output, _ = predictor.lstm(encoded)
        return output
    sequence = nn.utils.rnn.pack_padded_sequence(
        encoded,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, _ = predictor.lstm(sequence)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=padded_length,
    )
    return output


@torch.inference_mode()
def trace_duration_path(
    model: nn.Module,
    tokens: torch.Tensor,
    real_length: int,
    style_vector: torch.Tensor,
) -> DurationTrace:
    lengths = torch.tensor([real_length], device=tokens.device)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    bert = model.bert(tokens, attention_mask=(~mask).int())
    bert_encoded = model.bert_encoder(bert).transpose(-1, -2)
    style = style_vector[:, 128:]
    duration_encoded, stages = _run_conditioned_encoder(
        model.predictor.text_encoder,
        bert_encoded,
        style,
        lengths,
        mask,
    )
    stages["predictor.text_encoder.output"] = duration_encoded
    final = _run_final_lstm(
        model.predictor, duration_encoded, lengths, tokens.shape[1], packed=False
    )
    packed_final = _run_final_lstm(
        model.predictor, duration_encoded, lengths, tokens.shape[1], packed=True
    )
    logits = model.predictor.duration_proj(final)
    packed_logits = model.predictor.duration_proj(packed_final)
    continuous = torch.sigmoid(logits).sum(axis=-1)
    packed_continuous = torch.sigmoid(packed_logits).sum(axis=-1)
    stages.update(
        {
            "bert": bert,
            "bert_encoder": bert_encoded.transpose(-1, -2),
            "predictor.lstm": final,
            "predictor.lstm_packed_hack": packed_final,
            "duration_proj_logits": logits,
            "duration_proj_logits_packed_hack": packed_logits,
            "duration_continuous": continuous,
            "duration_continuous_packed_hack": packed_continuous,
            "duration_integer": torch.round(continuous).clamp(min=1),
            "duration_integer_packed_hack": torch.round(packed_continuous).clamp(min=1),
        }
    )
    return DurationTrace(stages)


@torch.inference_mode()
def trace_predictor_from_encoded(
    model: nn.Module,
    bert_encoded: torch.Tensor,
    style_vector: torch.Tensor,
) -> DurationTrace:
    real_length = bert_encoded.shape[1]
    lengths = torch.tensor([real_length], device=bert_encoded.device)
    mask = torch.zeros((1, real_length), dtype=torch.bool, device=bert_encoded.device)
    duration_encoded, stages = _run_conditioned_encoder(
        model.predictor.text_encoder,
        bert_encoded.transpose(-1, -2),
        style_vector[:, 128:],
        lengths,
        mask,
    )
    stages["predictor.text_encoder.output"] = duration_encoded
    return trace_from_duration_encoded(model, duration_encoded)


@torch.inference_mode()
def trace_from_duration_encoded(
    model: nn.Module,
    duration_encoded: torch.Tensor,
) -> DurationTrace:
    length = duration_encoded.shape[1]
    lengths = torch.tensor([length], device=duration_encoded.device)
    final = _run_final_lstm(
        model.predictor,
        duration_encoded,
        lengths,
        length,
        packed=True,
    )
    logits = model.predictor.duration_proj(final)
    continuous = torch.sigmoid(logits).sum(axis=-1)
    return DurationTrace(
        {
            "predictor.lstm": final,
            "duration_proj_logits": logits,
            "duration_continuous": continuous,
            "duration_integer": torch.round(continuous).clamp(min=1),
        }
    )
