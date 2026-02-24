from tinfer.executor.base import BaseExecutor
from tinfer.core.request import AudioChunkIPC, TTSRequestIPC, AudioChunk
from tinfer.process.worker import WorkerManager
from tinfer.process.shared_memory import SharedMemoryManager
from tinfer.process.protocol import MessageType, IPCProtocol
import torch
import threading
from typing import Any
from collections import defaultdict
import time
from dataclasses import replace
from tinfer.models.chunked import ChunkedModel
import numpy as np

class ProcessExecutor(BaseExecutor):
    def __init__(
        self,
        process_results,
        batch_size_per_device: dict[str, int],
        default_batch_size: int,
        devices: list[str] | None = None,
    ):
        self.process_results = process_results

        if devices is None or len(devices) == 0:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

        for device in devices:
            if device not in batch_size_per_device:
                batch_size_per_device[device] = default_batch_size

        self.devices = devices
        self.batch_size_per_device = batch_size_per_device
        self._workers: dict[int, Any] = {}
        self._model_to_worker: dict[str, int] = {}
        self._worker_round_robin = 0
        self._shm_manager = SharedMemoryManager()
        self._callback_thread: threading.Thread | None = None
        self._stop_callback = False
        self._pending_per_worker: dict[int, list[dict]] = defaultdict(list)
        self._load_args_per_worker: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)

    def _start_callback_thread(self):
        if self._callback_thread is not None and self._callback_thread.is_alive():
            return
            
        self._stop_callback = False
        def callback_loop():
            while not self._stop_callback:
                all_chunks: list[Any] = []
                for worker_id, worker in list(self._workers.items()):
                    chunks = worker.process_output_queue(self._shm_manager)
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        rid = getattr(chunk, "request_id", None)
                        cidx = getattr(chunk, "chunk_index", 0)
                        if rid is not None and self._pending_per_worker[worker_id]:
                            for i, p in enumerate(self._pending_per_worker[worker_id]):
                                if p.get("request_id") == rid and p.get("chunk_index") == cidx:
                                    self._pending_per_worker[worker_id].pop(i)
                                    break
                    if worker._process is not None and not worker._process.is_alive():
                        for p in self._pending_per_worker[worker_id]:
                            err = type("Result", (), {
                                "request_id": p["request_id"],
                                "nonce": p["nonce"],
                                "text_span": p.get("text_span", (0, 0)),
                                "chunk_index": p.get("chunk_index", 0),
                                "audio": np.zeros((0,), dtype=np.float32),
                                "sample_rate": 0,
                                "alignments": None,
                                "error": "Worker process on server terminated for some reason.",
                            })()
                            all_chunks.append(err)
                        self._pending_per_worker[worker_id] = []
                        device = self.devices[worker_id]
                        max_batch_size = self.batch_size_per_device.get(device, 1)
                        new_worker = WorkerManager(device, worker_id, max_batch_size)
                        new_worker._shm_manager = self._shm_manager
                        new_worker.run()
                        for model_id, args in self._load_args_per_worker.get(worker_id, {}).items():
                            new_worker.load_model(model_id, args["path"], voices_folder=args.get("voices_folder"), compile_models=args.get("compile_models", False))
                        self._workers[worker_id] = new_worker
                if all_chunks:
                    self.process_results(all_chunks)
                time.sleep(0.001)
        
        self._callback_thread = threading.Thread(target=callback_loop, daemon=True)
        self._callback_thread.start()

    def _model_load_register_util(self, model_id: str, device: str | None = None) -> None:
        if model_id in self._model_to_worker:
            return

        if device is None:
            worker_id = self._worker_round_robin % len(self.devices)
            self._worker_round_robin += 1
        else:
            worker_id = self.devices.index(device)

        if worker_id not in self._workers:
            device = self.devices[worker_id]
            max_batch_size = self.batch_size_per_device.get(device)
            worker = WorkerManager(device, worker_id, max_batch_size)
            worker.run()
            self._workers[worker_id] = worker

        self._model_to_worker[model_id] = worker_id

        return self._workers[worker_id]

    def load_model(self, model_id: str, path: str, device: str | None = None, voices_folder: str | None = None, compile_models: bool = False) -> None:
        worker = self._model_load_register_util(model_id, device)
        worker_id = self._model_to_worker[model_id]
        self._load_args_per_worker[worker_id][model_id] = {"path": path, "voices_folder": voices_folder, "compile_models": compile_models}
        worker.load_model(model_id, path, voices_folder=voices_folder, compile_models=compile_models)
        self._start_callback_thread()

    def register_model(self, model_id: str, model: Any, device: str | None = None, keep_in_main: bool = True) -> None:
        worker = self._model_load_register_util(model_id, device)
    
        state = model.get_state()
        state = self._shm_manager.serialize_recursive(state)
        worker.register_model(model_id, state)

        self._start_callback_thread()

    def send_to_process(self, items: list[TTSRequestIPC]) -> None:
        if not items:
            return
        self._start_callback_thread()
        items_by_worker: dict[int, list[TTSRequestIPC]] = defaultdict(list)
        
        for item in items:
            worker_id = self._model_to_worker.get(item.model_id)
            if worker_id is None:
                raise ValueError(f"Model {item.model_id} not found")
            
            serialized_context = self._shm_manager.serialize_recursive(item.context)
            item_with_shm = replace(item, context=serialized_context)
            
            items_by_worker[worker_id].append(item_with_shm)
        
        for worker_id, worker_items in items_by_worker.items():
            for item in worker_items:
                self._pending_per_worker[worker_id].append({
                    "request_id": item.request_id,
                    "nonce": item.nonce,
                    "text_span": getattr(item, "text_span", (0, 0)),
                    "chunk_index": getattr(item, "chunk_index", 0),
                })
            worker = self._workers[worker_id]
            worker.send_to_process(worker_items)
    
    def cancel_request(self, request_id: str) -> None:
        for worker in self._workers.values():
            worker.cancel_request(request_id)

    # def get_results(self) -> list[AudioChunkIPC]:
    #     all_results = []
    #     for worker in self._workers.values():
    #         results = worker.get_results()
    #         for result in results:
    #             all_results.append(result)

    #     for result in all_results:
    #         result.audio = self._shm_manager.deserialize_array(result.audio)
        
    #     return all_results

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._model_to_worker:
            raise ValueError(f"Model {model_id} not found")
        worker_id = self._model_to_worker[model_id]
        worker = self._workers[worker_id]
        worker.unload_model(model_id)
        del self._model_to_worker[model_id]
        if worker_id in self._load_args_per_worker:
            self._load_args_per_worker[worker_id].pop(model_id, None)


    def get_model_ids(self) -> list[str]:
        return list(self._model_to_worker.keys())

    def run(self) -> None:
        self._start_callback_thread()

    def stop(self) -> None:
        self._stop_callback = True
        if self._callback_thread and self._callback_thread.is_alive():
            self._callback_thread.join(timeout=1.0)
        for worker in self._workers.values():
            if worker._process and worker._process.is_alive():
                if worker._input_queue is not None:
                    worker._ipc_protocol.send_message(
                        worker._input_queue,
                        MessageType.SHUTDOWN,
                        {}
                    )
                worker._process.join(timeout=1.0)
                if worker._process.is_alive():
                    worker._process.terminate()
                    worker._process.join(timeout=2.0)
                    if worker._process.is_alive():
                        worker._process.kill()
                        worker._process.join()
        if self._shm_manager:
            self._shm_manager.cleanup()

    
