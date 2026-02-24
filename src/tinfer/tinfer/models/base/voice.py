from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class VoiceEncoder(ABC):
    @abstractmethod
    def encode(self, audio: np.ndarray, sample_rate: int):
        pass