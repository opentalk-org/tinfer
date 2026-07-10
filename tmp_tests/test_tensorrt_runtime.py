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

    def test_decoder_shape_bucket_uses_power_of_two_batch_and_frame_buckets(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import decoder_shape_bucket

        bucket = decoder_shape_bucket(batch_size=3, asr_frames=257, max_batch=16, max_frames=1024)

        self.assertEqual(bucket.batch_size, 4)
        self.assertEqual(bucket.asr_frames, 512)
        self.assertEqual(bucket.f0_frames, 1024)
        self.assertEqual(bucket.audio_samples, 307200)

    def test_decoder_shape_bucket_clips_to_configured_maximums(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import decoder_shape_bucket

        bucket = decoder_shape_bucket(batch_size=17, asr_frames=2049, max_batch=16, max_frames=1024)

        self.assertEqual(bucket.batch_size, 16)
        self.assertEqual(bucket.asr_frames, 1024)
        self.assertEqual(bucket.f0_frames, 2048)


class TensorRTPathTests(unittest.TestCase):
    def test_decoder_engine_path_is_stable(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import DecoderTRTEngineSpec

        spec = DecoderTRTEngineSpec(batch_size=8, asr_frames=512)

        self.assertEqual(spec.file_name, "decoder_b8_t512.engine")
        self.assertEqual(spec.onnx_name, "decoder_b8_t512.onnx")

    def test_decoder_engine_spec_parses_file_name(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import DecoderTRTEngineSpec

        spec = DecoderTRTEngineSpec.from_file_name("decoder_b16_t1024.engine")

        self.assertEqual(spec.batch_size, 16)
        self.assertEqual(spec.asr_frames, 1024)

    def test_missing_decoder_engine_raises_clear_error(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_runtime import TensorRTDecoderRunner

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "decoder_b1_t256.engine"):
                TensorRTDecoderRunner(Path(tmp_dir), batch_size=1, asr_frames=256)


if __name__ == "__main__":
    unittest.main()
