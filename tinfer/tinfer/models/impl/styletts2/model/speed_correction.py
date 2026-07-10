import os

BASE_CORRECTION = float(os.getenv("BASE_CORRECTION", "1.4"))
SPEED_CORRECTION_RATE = float(os.getenv("SPEED_CORRECTION_RATE", "0.00115"))


def baseline_speed_corrected_for_request(
    base_speed: float,
    phoneme_count: int,
) -> float:
    length = min(max(phoneme_count, 1), 300)
    correction = BASE_CORRECTION - SPEED_CORRECTION_RATE * length
    return base_speed * correction
