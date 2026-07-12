from types import MethodType
from typing import Any

import torch


def local_attention_mask(valid: torch.Tensor, radius: int) -> torch.Tensor:
    positions = torch.arange(valid.shape[-1], device=valid.device)
    local = (positions[:, None] - positions[None, :]).abs() <= radius
    query_valid = valid.bool().unsqueeze(-1)
    key_valid = valid.bool().unsqueeze(1)
    return (local.unsqueeze(0) & query_valid & key_valid).to(valid.dtype)


def local_bert_forward(
    bert: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    mask = local_attention_mask(attention_mask, bert.local_attention_radius)
    embeddings = bert.embeddings(input_ids)
    minimum = torch.finfo(embeddings.dtype).min
    additive_mask = torch.where(mask.bool(), 0.0, minimum).to(embeddings.dtype)
    encoded = bert.encoder(
        embeddings,
        additive_mask.unsqueeze(1),
        **kwargs,
    )
    return encoded[0]


def install_local_attention(bert: Any, radius: int) -> None:
    bert.global_forward = bert.forward
    bert.local_attention_radius = radius
    bert.forward = MethodType(local_bert_forward, bert)
