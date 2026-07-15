import torch

from tinfer.models.impl.styletts2.model.modules.tensorrt_export import onnx_istft20_inverse
from tools.styletts2_model_scripts.fourier import stft20


def test_stft20_matches_torch_for_fixed_source_window() -> None:
    torch.manual_seed(7)
    source = torch.randn(2, 105600)
    window = torch.hann_window(20)

    magnitude, phase = stft20(source, window)
    expected = torch.stft(
        source,
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=window,
        return_complex=True,
    )

    assert magnitude.shape == (2, 11, 21121)
    assert torch.allclose(magnitude, expected.abs(), atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.polar(magnitude, phase), expected, atol=2e-5, rtol=2e-5)


def test_fixed_istft20_matches_torch() -> None:
    torch.manual_seed(11)
    magnitude = torch.rand(2, 11, 257) + 0.1
    phase = torch.randn(2, 11, 257)
    window = torch.hann_window(20)

    audio = onnx_istft20_inverse(magnitude, phase, window).squeeze(1)
    expected = torch.istft(
        torch.polar(magnitude, phase),
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=window,
    )

    assert audio.shape == expected.shape
    assert torch.allclose(audio, expected, atol=2e-5, rtol=2e-5)
