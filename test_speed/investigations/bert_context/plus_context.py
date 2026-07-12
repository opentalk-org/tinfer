from types import MethodType

import torch
from torch import nn


def plus_context_forward(
    bert: nn.Module,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    assert tokens.shape[0] == 1, "BERT context intervention requires batch size one"
    assert bool(attention_mask.bool().all()), "Input must contain no padding"
    content_length = tokens.shape[1] - 1
    end = content_length + bert.context_extra_tokens
    continuation = bert.context_tokens[content_length:end].to(tokens.device)
    assert continuation.numel() == bert.context_extra_tokens, (
        "Passage lacks the requested genuine continuation"
    )
    extended_tokens = torch.cat((tokens, continuation.unsqueeze(0)), dim=1)
    extended_mask = torch.cat(
        (
            attention_mask,
            torch.ones(
                1,
                bert.context_extra_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ),
        dim=1,
    )
    output = bert.context_original_forward(
        extended_tokens,
        attention_mask=extended_mask,
    )
    return output[:, : tokens.shape[1]]


def install_plus_context(
    bert: nn.Module,
    continuation_tokens: torch.Tensor,
    extra_tokens: int,
) -> None:
    bert.context_original_forward = bert.forward
    bert.context_tokens = continuation_tokens.detach().cpu()
    bert.context_extra_tokens = extra_tokens
    bert.forward = MethodType(plus_context_forward, bert)
