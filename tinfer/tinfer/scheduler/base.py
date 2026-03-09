from abc import ABC, abstractmethod
from tinfer.core.request import TTSRequest

class Scheduler(ABC):
    @abstractmethod
    def schedule(self) -> list[tuple[TTSRequest, str]]:
        pass