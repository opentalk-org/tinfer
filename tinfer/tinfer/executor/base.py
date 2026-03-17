from abc import ABC, abstractmethod
from tinfer.core.request import TTSRequestIPC, AudioChunkIPC

class BaseExecutor(ABC):
    @abstractmethod
    def load_model(self, model_id: str, path: str, device: str | None = None) -> None:
        pass

    @abstractmethod
    def send_to_process(self, items: list[TTSRequestIPC]) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        pass