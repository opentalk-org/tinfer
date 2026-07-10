from __future__ import annotations

import time
from collections.abc import Callable


class FirstAudioLatencyTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started_at: float | None = None

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = self._clock()

    def consume_ms(self) -> int | None:
        if self._started_at is None:
            return None
        elapsed_ms = int(round((self._clock() - self._started_at) * 1000))
        self._started_at = None
        return elapsed_ms
