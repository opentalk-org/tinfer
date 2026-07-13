from contextlib import contextmanager
import os
import time
from typing import Any

import numpy as np
import torch

from tinfer.core.request import AlignmentItem
from tinfer.support.observability import get_logger

log = get_logger(__name__)

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def _find_leading_audio_samples(audio: np.ndarray, sample_rate: int) -> int:
    if audio.size == 0 or sample_rate <= 0:
        return 0

    frame_size = max(1, int(sample_rate * 0.02))
    starts = np.arange(0, audio.size, frame_size)
    if starts.size == 0:
        return 0

    audio64 = audio.astype(np.float64, copy=False)
    squared = audio64 * audio64
    frame_energy = np.add.reduceat(squared, starts)
    frame_lengths = np.minimum(frame_size, audio.size - starts)
    rms = np.sqrt(frame_energy / frame_lengths)
    threshold = max(0.01, float(np.percentile(rms, 80)) * 0.1)
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return 0

    first_frame = int(above[0])
    trim_samples = first_frame * frame_size
    return trim_samples if trim_samples >= int(sample_rate * 0.05) else 0


def _trim_leading_silence_and_shift_alignments(
    audio: np.ndarray,
    sample_rate: int,
    alignments: list[AlignmentItem],
) -> tuple[np.ndarray, list[AlignmentItem]]:
    trim_samples = _find_leading_audio_samples(audio, sample_rate)
    if trim_samples <= 0:
        return audio, alignments

    trim_ms = int(round(trim_samples / sample_rate * 1000.0))
    shifted = [
        AlignmentItem(
            item=item.item,
            char_start=item.char_start,
            char_end=item.char_end,
            start_ms=max(0, item.start_ms - trim_ms),
            end_ms=max(0, item.end_ms - trim_ms),
        )
        for item in alignments
    ]
    return audio[trim_samples:], shifted


def _alignment_items_for_debug_log(alignments: list[AlignmentItem]) -> list[dict[str, Any]]:
    return [
        {
            "item": item.item,
            "start_ms": item.start_ms,
            "end_ms": item.end_ms,
            "char_start": item.char_start,
            "char_end": item.char_end,
        }
        for item in alignments
    ]


@contextmanager
def timed_operation(name: str):
    profile_enabled = bool(os.getenv("TINFER_PROFILE"))
    if profile_enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        if profile_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log.debug("model_stage_profile", scope="model_stage", stage=name, elapsed_ms=elapsed_ms)
