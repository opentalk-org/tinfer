import unittest

import torch

from test_speed.duration_padding.local_attention_patch import local_attention_mask


class LocalAttentionMaskTests(unittest.TestCase):
    def test_masks_distant_and_padded_keys(self) -> None:
        valid = torch.tensor([[1, 1, 1, 1, 0]])

        mask = local_attention_mask(valid, radius=1)

        expected = torch.tensor(
            [[
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]]
        )
        torch.testing.assert_close(mask, expected)


if __name__ == "__main__":
    unittest.main()
