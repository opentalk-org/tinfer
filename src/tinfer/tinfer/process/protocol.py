from enum import Enum
from multiprocessing import Queue
from typing import Any

class MessageType(Enum):
    LOAD_MODEL = "load_model"
    REGISTER_MODEL = "register_model"
    UNLOAD_MODEL = "unload_model"
    CANCEL_REQUEST = "cancel_request"
    GET_RESULTS = "get_results"
    CLEANUP = "cleanup"
    SHUTDOWN = "shutdown"
    GENERATE_REQUEST = "generate_request"

class IPCProtocol:
    def send_message(self, queue: Queue, message_type: MessageType, data: Any) -> None:
        message = {
            "type": message_type.value,
            "data": data
        }
        queue.put(message)

    def receive_message(self, queue: Queue, timeout: float | None = None) -> dict[str, Any] | None:
        try:
            message = queue.get(timeout=timeout)
            return message
        except:
            return None