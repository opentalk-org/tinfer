import unittest

import torch

from tinfer.models.impl.styletts2.model.random_state import request_generator


class SeededGenerationTest(unittest.TestCase):
    def test_request_generators_are_repeatable_and_isolated(self) -> None:
        first = torch.randn(8, generator=request_generator(7, torch.device("cpu")))
        second = torch.randn(8, generator=request_generator(7, torch.device("cpu")))
        different = torch.randn(8, generator=request_generator(8, torch.device("cpu")))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))


if __name__ == "__main__":
    unittest.main()
