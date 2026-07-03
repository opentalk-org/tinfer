import math
import os
import re

BASE_SPEED = float(os.getenv("BASE_SPEED", "1.0"))
SPEED_CORRECTION_RATE = float(os.getenv("SPEED_CORRECTION_RATE", "1.0"))


def baseline_speed_corrected(base_speed: float, text: str) -> float:
    cleaned_text_length = len(cleanup_for_speed(text))
    return base_speed * speed_multiplier(cleaned_text_length)


def baseline_speed_corrected_for_request(
    base_speed: float,
    text: str,
    source_text: str | None,
    segment_index: int,
) -> float:
    correction_text = source_text if source_text is not None else text
    cleaned_text_length = len(cleanup_for_speed(correction_text))
    return base_speed * speed_multiplier(cleaned_text_length)


def cleanup_for_speed(text: str) -> str:
    normalized = re.sub(r"\s\s*", " ", text.lower())
    return re.sub(r"[^a-ząęćłóśźżń ]", "", normalized)


def compensated(length: int) -> float:
    return accumulated_speed_factor(length) / 4 + 0.8


def speed_multiplier(cleaned_text_length: int) -> float:
    correction = SPEED_CORRECTION_RATE * compensated(cleaned_text_length) + (1 - SPEED_CORRECTION_RATE)
    return correction * BASE_SPEED


def accumulated_speed_factor(length: int) -> float:
    if length < 1:
        return speed_activation(1)
    if length > 250:
        return speed_activation(250)
    return speed_activation(length)


def speed_activation(length: int) -> float:
    return math.atan(250) - math.atan(length / 60)
