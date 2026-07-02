from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TensorRTShapeTests(unittest.TestCase):
    def test_next_power_of_two_clips_to_bounds(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import next_power_of_two

        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(2), 2)
        self.assertEqual(next_power_of_two(3), 4)
        self.assertEqual(next_power_of_two(9), 16)
        self.assertEqual(next_power_of_two(9, maximum=8), 8)
        self.assertEqual(next_power_of_two(0, minimum=2), 2)


class TensorRTPathTests(unittest.TestCase):
    def test_engine_file_names_are_stable(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import (
            DynamicDecoderTRTEngineSpec,
            DynamicDiffusionTRTEngineSpec,
        )

        self.assertEqual(DynamicDecoderTRTEngineSpec().file_name, "decoder_dynamic.engine")
        self.assertEqual(DynamicDiffusionTRTEngineSpec(num_steps=10).file_name, "diffusion_dynamic_s10.engine")
        self.assertEqual(
            DynamicDiffusionTRTEngineSpec.from_file_name("diffusion_dynamic_s8.engine").num_steps,
            8,
        )

    def test_missing_decoder_engine_raises_clear_error(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import get_tensorrt_decoder_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "decoder_dynamic.engine"):
                get_tensorrt_decoder_runner(Path(tmp_dir))


if __name__ == "__main__":
    unittest.main()
