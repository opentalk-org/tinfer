import unittest

import torch
from torch import nn

from test_speed.investigations.bert_context.plus_context import install_plus_context


class RecordingBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_tokens = None
        self.received_mask = None

    def forward(self, tokens, attention_mask):
        self.received_tokens = tokens.clone()
        self.received_mask = attention_mask.clone()
        return tokens.unsqueeze(-1).float()


class PlusContextTests(unittest.TestCase):
    def test_appends_continuation_for_bert_and_crops_output(self) -> None:
        bert = RecordingBert()
        continuation = torch.arange(1, 20)
        install_plus_context(bert, continuation, extra_tokens=4)

        output = bert(
            torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3, dtype=torch.int),
        )

        self.assertEqual(bert.received_tokens.tolist(), [[0, 1, 2, 3, 4, 5, 6]])
        self.assertEqual(bert.received_mask.tolist(), [[1, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(output.squeeze(-1).tolist(), [[0.0, 1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
