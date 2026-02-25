from time import monotonic
from typing import Any
from collections import defaultdict
from tinfer.core.request import TTSRequestIPC

class WorkerScheduler:
    def __init__(self, max_batch_size: int):
        self._requests: dict[str, TTSRequestIPC] = {}
        self.max_batch_size = max_batch_size
    
    def add_request(self, request: TTSRequestIPC):
        self._requests[request.ipc_id] = request
    
    def remove_request(self, ipc_id: str):
        if ipc_id in self._requests:
            del self._requests[ipc_id]

    def schedule_batches(
        self,
        cancelled_requests: set[str]
    ) -> list[list[TTSRequestIPC]]:
        now = monotonic()
        
        for request in list(self._requests.values()):
            if request.request_id in cancelled_requests:
                self.remove_request(request.ipc_id)

        if not self._requests:
            return []

        requests_by_model: dict[str, list[TTSRequestIPC]] = defaultdict(list)
        for req in self._requests.values():
            requests_by_model[req.model_id].append(req)
        
        batches = []
        
        for model_id, model_requests in requests_by_model.items():
            sorted_requests = self.calculate_priority(model_requests)

            batch = sorted_requests[:self.max_batch_size]
            batches.append(batch)

        batches = self.calculate_batch_priority(batches)
        
        return batches

    def calculate_batch_priority(self, batches: list[list[TTSRequestIPC]]) -> list[list[TTSRequestIPC]]:
        def batch_score(batch: list[TTSRequestIPC]) -> float:

            return max(self.calculate_request_priority(req) for req in batch)
        
        sorted_batches = sorted(batches, key=batch_score, reverse=True)
        return sorted_batches

    def calculate_request_priority(self, request: TTSRequestIPC) -> float:
        score = 0.0

        # TODO: double check which exacly can't be None
        if request.start_time is not None and request.collected_time is not None and monotonic() - request.start_time > request.collected_time:
            score += 1e9

        now = monotonic()
        elapsed_time = now - request.created_at
        score += elapsed_time

        return score

    def calculate_priority(self, requests: list[TTSRequestIPC]) -> float:
        scores = {}

        for request in requests:
            scores[request.ipc_id] = self.calculate_request_priority(request)

        sorted_requests = sorted(requests, key=lambda r: scores[r.ipc_id], reverse=True)
        return sorted_requests
        


