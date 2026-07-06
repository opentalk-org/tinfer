from tinfer.models.impl.styletts2.model import speed_correction


def test_speed_correction_uses_phoneme_count(monkeypatch):
    monkeypatch.setattr(speed_correction, "BASE_CORRECTION", 1.0)
    monkeypatch.setattr(speed_correction, "SPEED_CORRECTION_RATE", 0.001)

    corrected = speed_correction.baseline_speed_corrected_for_request(2.0, 40)

    assert corrected == 1.92


def test_speed_correction_clamps_phoneme_count(monkeypatch):
    monkeypatch.setattr(speed_correction, "BASE_CORRECTION", 1.0)
    monkeypatch.setattr(speed_correction, "SPEED_CORRECTION_RATE", 0.001)

    short = speed_correction.baseline_speed_corrected_for_request(1.0, 0)
    long = speed_correction.baseline_speed_corrected_for_request(1.0, 300)

    assert short == 0.999
    assert long == 0.75
