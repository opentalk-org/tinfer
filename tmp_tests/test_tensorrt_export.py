from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from tinfer.models.impl.styletts2.model.modules.istftnet import TorchSTFT


class TensorRTExportTests(unittest.TestCase):
    def test_onnx_istft20_matches_torch_istft_for_model_window(self):
        from tinfer.models.impl.styletts2.model.modules.tensorrt_export import onnx_istft20_inverse

        torch.manual_seed(1234)
        stft = TorchSTFT(filter_length=20, hop_length=5, win_length=20)
        audio = torch.randn(2, 240)
        magnitude, phase = stft.transform(audio)

        expected = stft.inverse(magnitude, phase)
        actual = onnx_istft20_inverse(magnitude, phase, stft.window)

        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(torch.max(torch.abs(actual - expected)).item(), 1e-4)

    def test_torch_stft_inverse_exports_without_complex_istft_in_trt_mode(self):
        class InverseOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stft = TorchSTFT(filter_length=20, hop_length=5, win_length=20)

            def forward(self, magnitude, phase):
                return self.stft.inverse(magnitude, phase)

        model = InverseOnly()
        magnitude = torch.rand(1, 11, 16)
        phase = torch.rand(1, 11, 16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "inverse.onnx")
            with patch.dict(os.environ, {"TINFER_TRT_EXPORT": "1"}):
                torch.onnx.export(
                    model,
                    (magnitude, phase),
                    onnx_path,
                    input_names=["magnitude", "phase"],
                    output_names=["audio"],
                    opset_version=20,
                    dynamo=False,
                )

            self.assertTrue(os.path.exists(onnx_path))

    def test_decoder_forward_with_precomputed_har_matches_forward(self):
        from torch.nn.utils import remove_weight_norm

        from tinfer.models.impl.styletts2.model.modules.istftnet import Decoder
        from tinfer.models.impl.styletts2.model.modules.tensorrt_export import DecoderTRTExportModule

        torch.manual_seed(4321)
        decoder = Decoder().eval()
        for module in decoder.modules():
            try:
                remove_weight_norm(module)
            except ValueError:
                pass

        asr = torch.randn(1, 512, 8)
        f0 = torch.rand(1, 16) * 120.0 + 80.0
        noise = torch.randn(1, 16)
        style = torch.randn(1, 64)

        with torch.no_grad():
            torch.manual_seed(99)
            har = decoder.generator._preprocess_f0(f0)
            actual = decoder.forward_with_har(asr, f0, noise, style, har)
            wrapped = DecoderTRTExportModule(decoder)(asr, f0, noise, style, har)

            torch.manual_seed(99)
            expected = decoder(asr, f0, noise, style)

        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(torch.max(torch.abs(actual - expected)).item(), 1e-4)
        self.assertLess(torch.max(torch.abs(wrapped - expected)).item(), 1e-4)

    def test_adain_uses_manual_instance_norm_in_trt_export_mode(self):
        from tinfer.models.impl.styletts2.model.modules.decoder_blocks import AdaIN1d

        class FailingNorm(torch.nn.Module):
            def forward(self, x):
                raise AssertionError("nn.InstanceNorm1d should not run in TRT export mode")

        adain = AdaIN1d(style_dim=4, num_features=3)
        adain.norm = FailingNorm()
        x = torch.randn(2, 3, 5)
        style = torch.randn(2, 4)

        with patch.dict(os.environ, {"TINFER_TRT_EXPORT": "1"}):
            out = adain(x, style)

        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
