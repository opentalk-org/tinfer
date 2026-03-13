from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from tinfer.core.request import AlignmentItem, AlignmentType


class AlignmentParser(ABC):
    @abstractmethod
    def get_native_alignment_type(self) -> AlignmentType:

        pass
    







