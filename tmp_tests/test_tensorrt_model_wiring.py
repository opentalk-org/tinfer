from __future__ import annotations

import tempfile
import unittest

import torch

from tinfer.models.impl.styletts2.model.model import StyleTTS2
from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import DecoderShapeBucket


class TensorRTModelWiringTests(unittest.TestCase):
    def test_get_trt_decoder_runner_requires_engine_file(self):
        model = StyleTTS2(device="cuda")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model._trt_engine_dir = tmp_dir
            model._trt_decoder_runners = {}

            bucket = DecoderShapeBucket(batch_size=1, asr_frames=256, f0_frames=512, audio_samples=153600)

            with self.assertRaisesRegex(FileNotFoundError, "decoder_b1_t256.engine"):
                model._get_trt_decoder_runner(bucket)

    def test_pad_or_clip_to_size_repeats_last_item_and_clips(self):
        model = StyleTTS2(device="cuda")
        tensor = torch.tensor([[[1.0, 2.0, 3.0]]])

        padded = model._pad_or_clip_to_size(tensor, 5, dim=-1)
        clipped = model._pad_or_clip_to_size(tensor, 2, dim=-1)

        self.assertEqual(padded.tolist(), [[[1.0, 2.0, 3.0, 3.0, 3.0]]])
        self.assertEqual(clipped.tolist(), [[[1.0, 2.0]]])


if __name__ == "__main__":
    unittest.main()
