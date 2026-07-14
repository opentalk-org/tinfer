from pathlib import Path

import numpy as np
import pytest
import torch

from tools.styletts2_model_scripts.convert_voices import convert_voices, mono_waveform, validate_voice_vector


def test_mono_waveform_averages_soundfile_channel_axis() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    assert np.array_equal(mono_waveform(stereo), np.array([0.0, 0.375], dtype=np.float32))
    assert np.array_equal(mono_waveform(stereo[:, 0]), stereo[:, 0])


def test_validate_voice_vector_requires_256_finite_float_values() -> None:
    vector = validate_voice_vector(torch.arange(256, dtype=torch.float32).reshape(1, 256))
    assert vector.shape == (256,)
    assert vector.dtype == np.float32

    with pytest.raises(ValueError, match="256"):
        validate_voice_vector(torch.zeros(255))
    invalid = torch.zeros(256)
    invalid[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        validate_voice_vector(invalid)


def test_convert_voices_rejects_duplicate_output_names_before_model_loading(tmp_path: Path) -> None:
    first = tmp_path / "a" / "voice.wav"
    second = tmp_path / "b" / "voice.wav"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="duplicate voice name"):
        convert_voices(tmp_path / "model", (first, second), tmp_path / "voices")
