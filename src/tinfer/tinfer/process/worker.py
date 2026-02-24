from tinfer.process.shared_memory import SharedMemoryManager
from tinfer.process.protocol import IPCProtocol, MessageType
from tinfer.core.request import TTSRequestIPC, AudioChunkIPC, AudioChunk, Alignment, AlignmentItem, AlignmentType
from tinfer.models.base.model import IntermediateRepresentation
from tinfer.scheduler.worker_scheduler import WorkerScheduler
from multiprocessing import get_context
from typing import Any
from tinfer.models.registry import get_model_class
import asyncio
import numpy as np
import queue as queue_module
import time
import multiprocessing
import signal
import torch
from tinfer.utils.audio_encoder import AudioFormat, DefaultAudioEncoder

_mp_context = get_context('spawn')
Process = _mp_context.Process
Queue = _mp_context.Queue
class WorkerProcess(Process):
    def __init__(self, device: str, worker_id: int, input_queue: Queue, output_queue: Queue, max_batch_size: int):
        super().__init__()
        self.device = device
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.max_batch_size = max_batch_size
        
        self.shm_manager = SharedMemoryManager()
        self.ipc_protocol = IPCProtocol()
        self.models: dict[str, Any] = {}
        self.request_queue: queue_module.Queue[TTSRequestIPC] | None = None
        self.cancelled_requests: set[str] = set()
        self.scheduler = WorkerScheduler(max_batch_size)
        self.audio_encoder = DefaultAudioEncoder()

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.request_queue = queue_module.Queue()
            while True:
                processed_messages = self._process_messages()
                
                batches = self.scheduler.schedule_batches(self.cancelled_requests)
                
                for batch in batches:
                    self._process_batch(batch)
                
                if not processed_messages and self.request_queue.empty():
                    time.sleep(0.010)
        finally:
            self.shm_manager.cleanup()
    
    def _process_messages(self) -> bool:
        processed_messages = False
        
        while True:
            message = self.ipc_protocol.receive_message(self.input_queue, timeout=0.0)
            if message is None:
                break
            
            processed_messages = True
            msg_type = MessageType(message["type"])
            data = message["data"]
            
            if msg_type == MessageType.LOAD_MODEL:
                self._handle_load_model(data)
            elif msg_type == MessageType.REGISTER_MODEL:
                self._handle_register_model(data)
            elif msg_type == MessageType.UNLOAD_MODEL:
                self._handle_unload_model(data)
            elif msg_type == MessageType.CANCEL_REQUEST:
                self._handle_cancel_request(data)
            elif msg_type == MessageType.SHUTDOWN:
                self.shm_manager.cleanup()
                raise SystemExit(0)
            elif msg_type == MessageType.GENERATE_REQUEST:
                self._handle_generate_request(data)
        
        return processed_messages
    
    def _handle_load_model(self, data: dict) -> None:
        model_id = data["model_id"]
        path = data["path"]
        compile_models = data["compile_models"]
        voices_folder = data.get("voices_folder")
        model_class = get_model_class("styletts2") # TODO: handle other models in future
        model = model_class()
        model.load(path, voices_folder=voices_folder, compile_model=compile_models)
        model.max_batch_size = self.max_batch_size
        self.models[model_id] = model
    
    def _handle_register_model(self, data: dict) -> None:
        model_id = data["model_id"]
        state = data["state"]
        model_class = get_model_class("styletts2") # TODO: handle other models in future
        model = model_class()
        deserialized_state = self.shm_manager.deserialize_recursive(state)
        model.load_from_state(deserialized_state)
        model.max_batch_size = self.max_batch_size
        self.models[model_id] = model
    
    def _handle_unload_model(self, data: dict) -> None:
        model_id = data["model_id"]
        del self.models[model_id]
    
    def _handle_cancel_request(self, data: dict) -> None:
        request_id = data["request_id"]
        self.cancelled_requests.add(request_id)
    
    def _handle_generate_request(self, data: dict) -> None:
        request_ipc: TTSRequestIPC = data["request"]
        if request_ipc.request_id in self.cancelled_requests:
            self.cancelled_requests.remove(request_ipc.request_id)
        self.scheduler.add_request(request_ipc)
    
    def _process_batch(self, batch: list[TTSRequestIPC]) -> None:
        if not batch:
            return
        
        model_id = batch[0].model_id
        model = self.models.get(model_id)
        
        try:
            texts, contexts, params_list, request_metadata = self._prepare_batch_data(batch)
            
            if not texts:
                self._mark_batch_complete(batch)
                return
            
            start = time.monotonic()
            with torch.no_grad():
                results = self._execute_batch(model, texts, contexts, params_list, request_metadata)
            torch.cuda.synchronize()
            end = time.monotonic()
            # print(f"MODEL_GENERATE_TIME: {end - start}")
            self._send_batch_results(results, request_metadata)
        except Exception as e:
            for req in batch:
                metadata = {
                    "request_id": req.request_id,
                    "chunk_index": req.chunk_index,
                    "text_span": req.text_span,
                    "nonce": req.nonce,
                }
                self._send_error_result(metadata, str(e))
        finally:
            self._mark_batch_complete(batch)

    def _prepare_batch_data(self, batch: list[TTSRequestIPC]) -> tuple[list[str], list[dict], list[dict], list[dict]]:
        texts = []
        contexts = []
        params_list = []
        request_metadata = []
        
        for req in batch:
            context = self.shm_manager.deserialize_recursive(req.context)
            texts.append(req.text)
            contexts.append(context)
            params_list.append(req.params)
            request_metadata.append({
                "request_id": req.request_id,
                "ipc_id": req.ipc_id,
                "chunk_index": req.chunk_index,
                "text_span": req.text_span,
                "text": req.text,
                "alignment_type": req.alignment_type,
                "nonce": req.nonce,
                "target_sample_rate": req.target_sample_rate,
                "target_encoding": req.target_encoding,
            })
        
        return texts, contexts, params_list, request_metadata
    
    def _execute_batch(self, model: Any, texts: list[str], contexts: list[dict], params_list: list[dict], request_metadata: list[dict]) -> list[IntermediateRepresentation] | None:
        # TODO support for stream generate
        if len(texts) > 1:
            return model.generate_batch(texts, contexts, params_list, request_metadata)
        else:
            return [model.generate(texts[0], contexts[0], params_list[0], request_metadata[0])]
    
    # def _execute_stream(self, model: Any, method: str, texts: list[str], contexts: list[dict], params_list: list[dict]) -> list[IntermediateRepresentation]:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         if method == "stream_batch":
    #             async_gen = model.stream_batch(texts, contexts, params_list)
    #         else:
    #             async_gen = model.stream(texts[0], contexts[0], params_list[0])
            
    #         results = []
    #         async def collect_results():
    #             async for item in async_gen:
    #                 if isinstance(item, list):
    #                     for ir in item:
    #                         if isinstance(ir, IntermediateRepresentation):
    #                             results.append(ir)
    #                 elif isinstance(item, IntermediateRepresentation):
    #                     results.append(item)
            
    #         loop.run_until_complete(collect_results())
    #         return results
    #     finally:
    #         loop.close()
    
    def _build_alignments(self, ir: IntermediateRepresentation) -> Alignment | None:        
        word_aligns = ir.metadata["word_alignments"]
        if not isinstance(word_aligns, list) or len(word_aligns) == 0:
            return None
        
        items = []
        for item in word_aligns:
            if isinstance(item, dict):
                items.append(AlignmentItem(
                    item=item.get("item", item.get("word", "")),
                    char_start=item.get("char_start", 0),
                    char_end=item.get("char_end", 0),
                    start_ms=item.get("start_ms", 0),
                    end_ms=item.get("end_ms", 0)
                ))
            elif hasattr(item, "item") or hasattr(item, "word"):
                items.append(item)
        
        if not items:
            return None
        
        alignment_type_str = ir.metadata.get("alignment_type", "word")
        alignment_type = AlignmentType(alignment_type_str)
        
        alignments = Alignment()
        alignments.items = items
        alignments.type_ = alignment_type
        return alignments
    
    def _send_error_result(self, metadata: dict, message: str) -> None:
        minimal_audio = np.zeros((1,), dtype=np.float32)
        audio_ref = self.shm_manager.serialize_array(minimal_audio)
        chunk_ipc = AudioChunkIPC(
            request_id=metadata["request_id"],
            audio=audio_ref,
            sample_rate=0,
            text_span=metadata["text_span"],
            alignments=None,
            chunk_index=metadata["chunk_index"],
            nonce=metadata["nonce"],
            error=message,
        )
        self.ipc_protocol.send_message(
            self.output_queue,
            MessageType.GET_RESULTS,
            {"result": chunk_ipc}
        )

    def _send_batch_results(self, results: list[IntermediateRepresentation], request_metadata: list[dict]) -> None:
        for i, ir in enumerate(results):

            metadata = request_metadata[i]
            request_id = metadata["request_id"]
            
            audio_data = ir.data
            sample_rate = ir.sample_rate
            
            target_encoding = metadata.get("target_encoding")
            target_sample_rate = metadata.get("target_sample_rate")
            
            if target_encoding is not None:
                if target_sample_rate is not None:
                    sample_rate = target_sample_rate
                else:
                    sample_rate = self.audio_encoder.get_sample_rate(target_encoding)
                
                audio_bytes = self.audio_encoder.encode(audio_data, ir.sample_rate, target_encoding)
                audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
            
            audio_ref = self.shm_manager.serialize_array(audio_data)
            alignments = self._build_alignments(ir)

            chunk_ipc = AudioChunkIPC(
                request_id=request_id,
                audio=audio_ref,
                sample_rate=sample_rate,
                text_span=metadata["text_span"],
                alignments=alignments,
                chunk_index=metadata["chunk_index"],
                nonce=metadata["nonce"],
            )
            
            self.ipc_protocol.send_message(
                self.output_queue,
                MessageType.GET_RESULTS,
                {"result": chunk_ipc}
            )
    
    def _mark_batch_complete(self, batch: list[TTSRequestIPC]) -> None:
        for req in batch:
            self.scheduler.remove_request(req.ipc_id)
class WorkerManager:
    def __init__(self, device: str, workder_id: int, max_batch_size: int):
        self.device = device
        self.workder_id = workder_id
        self.max_batch_size = max_batch_size
        self._input_queue: Queue | None = None
        self._output_queue: Queue | None = None
        self._process: Process | None = None
        self._ipc_protocol = IPCProtocol()

    def _start_process(self):
        if self._process is not None and self._process.is_alive():
            return
            
        self._input_queue = Queue()
        self._output_queue = Queue()
        self._process = WorkerProcess(
            self.device, self.workder_id, self._input_queue, self._output_queue, self.max_batch_size
        )
        self._process.start()

    def run(self):
        self._start_process()

    def load_model(self, model_id: str, path: str, voices_folder: str | None = None, compile_models: bool = False):
        self._start_process()
        message_data = {"model_id": model_id, "path": path, "voices_folder": voices_folder, "compile_models": compile_models}
        self._ipc_protocol.send_message(
            self._input_queue,
            MessageType.LOAD_MODEL,
            message_data
        )

    def register_model(self, model_id: str, state: dict):
        self._start_process()
        serialized_state = self._shm_manager.serialize_recursive(state)
        self._ipc_protocol.send_message(
            self._input_queue,
            MessageType.REGISTER_MODEL,
            {"model_id": model_id, "state": serialized_state}
        )

    def unload_model(self, model_id: str):
        self._ipc_protocol.send_message(
            self._input_queue,
            MessageType.UNLOAD_MODEL,
            {"model_id": model_id}
        )

    def send_to_process(self, items: list[TTSRequestIPC]):
        self._start_process()
        
        for item in items:
            self._ipc_protocol.send_message(
                self._input_queue,
                MessageType.GENERATE_REQUEST,
                {"request": item}
            )

    def cancel_request(self, request_id: str):
        self._ipc_protocol.send_message(
            self._input_queue,
            MessageType.CANCEL_REQUEST,
            {"request_id": request_id}
        )

    def get_results(self) -> list[AudioChunkIPC]:
        results = []
        while True:
            message = self._ipc_protocol.receive_message(self._output_queue, timeout=0.0)
            if message is None:
                break
                
            if message["type"] == MessageType.GET_RESULTS.value:
                chunk_data = message["data"]["result"]
                audio = chunk_data["audio"]

                chunk_with_audio = AudioChunkIPC(
                    request_id=chunk_data["request_id"],
                    audio=audio,
                    sample_rate=chunk_data["sample_rate"],
                    text_span=chunk_data["text_span"],
                    alignments=chunk_data["alignments"],
                    chunk_index=chunk_data["chunk_index"],
                    nonce=chunk_data["nonce"],
                    error=chunk_data.get("error") if isinstance(chunk_data, dict) else getattr(chunk_data, "error", None),
                )
                results.append(chunk_with_audio)
                
        return results

    def process_output_queue(self, shm_manager: SharedMemoryManager) -> list[AudioChunk]:
        if self._output_queue is None:
            return []
        
        chunks: list[AudioChunk] = []
        while True:
            message = self._ipc_protocol.receive_message(self._output_queue, timeout=0.0)
            if message is None:
                break
            
            if message["type"] == MessageType.GET_RESULTS.value:
                chunk_data = message["data"]["result"]
                audio_ref = chunk_data.audio
                audio_array = shm_manager.deserialize_array(audio_ref)
                chunk_data.audio = audio_array
                chunks.append(chunk_data)
        
        return chunks

    