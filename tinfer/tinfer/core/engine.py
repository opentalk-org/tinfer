import asyncio
import heapq
import threading
import uuid
from time import monotonic
from typing import Any

import numpy as np
import yaml

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.request import (
    Alignment,
    AlignmentItem,
    AudioChunk,
    AudioChunkIPC,
    StreamParams,
    TTSRequest,
    TTSRequestIPC,
)
from tinfer.core.stream import TTSStream
from tinfer.executor.process_executor import ProcessExecutor
from tinfer.support.errors import InferenceError
from tinfer.support.observability import get_logger
from tinfer.utils.audio_encoder import AudioFormat, parse_output_format
from tinfer.utils.text_chunker import TextChunker

log = get_logger(__name__)

_STREAM_PARAM_KEYS = frozenset({
    "chunk_length_schedule",
    "timeout_trigger_ms",
    "alignment_type",
    "target_sample_rate",
    "target_encoding",
    "tts_params",
})


class StreamingTTS:
    def __init__(self, config: StreamingTTSConfig, auto_run: bool = True):
        self.config = config

        self._requests: dict[str, TTSRequest] = {}

        self.timeout_thread = None
        self._timeout_events: list[tuple[float, int, str, float]] = []
        self._timeout_sequence = 0
        self._timeout_condition = threading.Condition()
        self._scheduler_lock = threading.RLock()
        self._stop_timeout = False

        self.executor = ProcessExecutor(
            self.process_results,
            config.batch_size_per_device,
            config.default_batch_size,
            config.devices,
        )
        self._chunker = TextChunker()

        if auto_run:
            self.run()

    @classmethod
    def from_config(cls, config: StreamingTTSConfig):
        if isinstance(config, dict):
            config = StreamingTTSConfig(**config)
        else:
            with open(config) as f:
                config_dict = yaml.safe_load(f)
            config = StreamingTTSConfig(**config_dict)
        return cls(config)

    async def generate_full(self, model_id: str, voice_id: str, text: str, params: StreamParams | dict[str, Any]):
        stream = self.create_stream(model_id, voice_id, params)

        stream.add_text(text)
        stream.force_generate()

        try:
            audio_chunks = []
            async for chunk in stream.pull_audio():
                audio_chunks.append(chunk)
            return self._merge_audio_chunks(audio_chunks)
        finally:
            stream.close()

    async def generate_full_batch(self, model_id: str, voice_id: str, texts: list[str], params: list[StreamParams | dict[str, Any]]):
        return await asyncio.gather(
            *[
                self.generate_full(model_id, voice_id, text, params_dict)
                for text, params_dict in zip(texts, params)
            ]
        )

    def _merge_audio_chunks(self, chunks: list[AudioChunk]) -> AudioChunk:
        if not chunks:
            return AudioChunk(audio=np.zeros((0,), dtype=np.float32), sample_rate=0)
        for c in chunks:
            if getattr(c, "error", None):
                raise InferenceError(c.error)
        sample_rate = chunks[0].sample_rate
        for c in chunks[1:]:
            if c.sample_rate != sample_rate:
                raise ValueError("Cannot merge audio chunks with different sample rates")

        audio = np.concatenate([c.audio for c in chunks], axis=-1) if len(chunks) > 1 else chunks[0].audio

        merged_alignment = None
        merged_items: list[AlignmentItem] = []
        alignment_type = None

        text_end = 0
        time_offset_ms = 0

        for c in chunks:
            if c.alignments is not None:
                if alignment_type is None and hasattr(c.alignments, "type_"):
                    alignment_type = c.alignments.type_
                if hasattr(c.alignments, "items") and c.alignments.items:
                    for it in c.alignments.items:
                        merged_items.append(
                            AlignmentItem(
                                item=it.item,
                                char_start=it.char_start + c.text_span[0],
                                char_end=it.char_end + c.text_span[0],
                                start_ms=it.start_ms + time_offset_ms,
                                end_ms=it.end_ms + time_offset_ms,
                            )
                        )

            text_end = max(text_end, int(c.text_span[1]))

            raw_len = int(c.audio.shape[-1]) if hasattr(c.audio, "shape") and len(c.audio.shape) > 0 else int(len(c.audio))
            num_samples = raw_len // 2 if hasattr(c.audio, "dtype") and c.audio.dtype == np.uint8 else raw_len
            time_offset_ms += int(round((num_samples / sample_rate) * 1000.0)) if sample_rate else 0

        if merged_items:
            merged_alignment = Alignment()
            merged_alignment.items = merged_items
            if alignment_type is not None:
                merged_alignment.type_ = alignment_type

        return AudioChunk(
            audio=audio,
            sample_rate=sample_rate,
            chunk_index=0,
            text_span=(0, text_end),
            alignments=merged_alignment,
        )

    def _close_stream(self, request_id: str):
        if request_id not in self._requests:
            return

        self._cancel_request(request_id)
        self._requests.pop(request_id)
        log.info("stream_closed", request_id=request_id, active_streams=len(self._requests))

    def load_model(self, model_id: str, model_path: str, voices_folder: str | None = None):
        log.info("engine_model_loading", model_id=model_id, path=model_path, voices_folder=voices_folder)
        self.executor.load_model(
            model_id,
            model_path,
            voices_folder=voices_folder,
            compile_models=self.config.compile_models,
            runtime_engine=self.config.runtime_engine,
        )

    def register_model(self, model_id: str, model, device: str | None = None, keep_in_main: bool = True):
        if not hasattr(model, '_loaded') or not model._loaded:
            raise ValueError(f"Model must be loaded before registration. Call model.load(path) first.")
        self.executor.register_model(model_id, model, device, keep_in_main)
        if not keep_in_main:
            if hasattr(model, '_model'):
                model._model = None
            if hasattr(model, '_loaded'):
                model._loaded = False

    def unload_model(self, model_id: str):
        self.executor.unload_model(model_id)
        log.info("engine_model_unloaded", model_id=model_id)

    def get_model_ids(self) -> list[str]:
        return self.executor.get_model_ids()

    def get_voice_ids(self, model_id: str) -> list[str]:
        return self.executor.get_voice_ids(model_id)

    def run(self):
        self._stop_timeout = False
        self.timeout_thread = threading.Thread(target=self.run_timeout_loop, daemon=True)
        self.timeout_thread.start()

    def schedule_timeout(self, request: TTSRequest) -> None:
        assert request.generation_window_started_at is not None, "cannot schedule a timeout without a generation window"
        started_at = request.generation_window_started_at
        deadline = started_at + request.timeout_trigger_ms / 1000.0
        with self._timeout_condition:
            self._timeout_sequence += 1
            heapq.heappush(
                self._timeout_events,
                (deadline, self._timeout_sequence, request.request_id, started_at),
            )
            self._timeout_condition.notify()

    def run_timeout_loop(self):
        while True:
            with self._timeout_condition:
                while not self._stop_timeout:
                    if not self._timeout_events:
                        self._timeout_condition.wait()
                        continue
                    deadline, _sequence, request_id, started_at = self._timeout_events[0]
                    wait_time = deadline - monotonic()
                    if wait_time > 0:
                        self._timeout_condition.wait(timeout=wait_time)
                        continue
                    heapq.heappop(self._timeout_events)
                    break
                if self._stop_timeout:
                    return

            request = self._requests.get(request_id)
            if request is not None and request.generation_window_started_at == started_at:
                self.signal_input()

    def process_results(self, results: list[AudioChunkIPC]):
        with self._scheduler_lock:
            self._process_results_locked(results)
            self._signal_input_locked()

    def _process_results_locked(self, results: list[AudioChunkIPC]):
        for result in results:
            request_id = result.request_id
            if request_id not in self._requests:
                continue
            
            request = self._requests[request_id]

            if result.nonce != request.nonce:
                continue

            if request.first_audio_at is None:
                request.first_audio_at = monotonic()
            
            error = getattr(result, "error", None)
            chunk = AudioChunk(
                audio=result.audio,
                sample_rate=result.sample_rate,
                text_span=result.text_span,
                alignments=result.alignments,
                chunk_index=result.chunk_index,
                error=error,
            )
            result_context = getattr(result, "context", None)
            if result_context is not None:
                request.set_state(result_context)
            request.audio_queue.put(chunk)
            request.pending_chunks -= 1
            log.debug(
                "audio_chunk_queued",
                request_id=request_id,
                chunk_index=result.chunk_index,
                pending_chunks=request.pending_chunks,
                audio_queue_size=request.audio_queue.qsize(),
                has_error=error is not None,
            )

            if result.error is None:
                request.collected_time += len(result.audio) / result.sample_rate
            else:
                request.collected_time = 0.0
                request.start_time = None

    def create_stream(self, model_id: str, voice_id: str, params: StreamParams | dict[str, Any]) -> TTSStream:
        request_id = str(uuid.uuid4())
        params = dict(params)
        tts_params = dict(params.pop("tts_params", {}))
        target_sample_rate = params.pop("target_sample_rate", None)
        target_encoding = params.pop("target_encoding", None)

        request_kwargs = {
            "request_id": request_id,
            "model_id": model_id,
            "voice_id": voice_id,
            "tts_params": tts_params,
            "nonce": str(uuid.uuid4()),
            "chunk_length_schedule": self.config.default_chunk_schedule.copy(),
            "timeout_trigger_ms": self.config.default_timeout_ms,
            "alignment_type": self.config.default_alignment_type,
            "target_sample_rate": target_sample_rate,
            "target_encoding": target_encoding,
        }
        for key in _STREAM_PARAM_KEYS:
            if key in ("target_sample_rate", "target_encoding", "tts_params"):
                continue
            if key in params:
                val = params[key]
                if key == "chunk_length_schedule" and isinstance(val, list):
                    val = list(val)
                request_kwargs[key] = val

        request = TTSRequest(**request_kwargs)
        self._requests[request_id] = request

        stream = TTSStream(request, self)

        return stream

    def _cancel_request(self, request_id: str):
        with self._scheduler_lock:
            self._cancel_request_locked(request_id)

    def _cancel_request_locked(self, request_id: str):
        if request_id not in self._requests:
            return

        self.executor.cancel_request(request_id)

        request = self._requests[request_id]
        request.text_buffer = ""
        request.text_committed_pos = 0
        request.generation_window_started_at = None
        request.prepared_chunks.clear()
        request.force_next_generation = False
        request.pending_chunks = 0
        request.nonce = str(uuid.uuid4())
        request.audio_queue.queue.clear()
        log.info("stream_cancelled", request_id=request_id)

    def add_text(self, request: TTSRequest, text: str) -> None:
        if not text.strip():
            return
        with self._scheduler_lock:
            starts_generation_window = request.append_text(text)
            if starts_generation_window:
                self.schedule_timeout(request)
            self._signal_input_locked()

    def force_generate(self, request: TTSRequest) -> None:
        with self._scheduler_lock:
            if not request.get_pending_text().strip():
                return
            request.force_next_generation = True
            self._signal_input_locked()

    def signal_input(self):
        with self._scheduler_lock:
            self._signal_input_locked()

    def _signal_input_locked(self):
        to_send = []
        requests_snapshot = list(self._requests.values())
        for request in requests_snapshot:
            
            single_chunk = True # TODO: get from model

            if single_chunk and request.pending_chunks > 0:
                continue

            chunks = self._chunker.get_chunks(request, single_chunk=single_chunk)

            context = request.get_state()
            context["voice_id"] = request.voice_id

            for chunk in chunks:

                text_chunk, current_chunk_index, is_final, text_span = chunk

                target_sample_rate = request.target_sample_rate
                target_encoding = request.target_encoding
                if isinstance(target_encoding, str):
                    target_encoding = parse_output_format(target_encoding)
                elif target_encoding is not None and not isinstance(target_encoding, AudioFormat):
                    target_encoding = None

                item = TTSRequestIPC(
                    request_id=request.request_id,
                    ipc_id=str(uuid.uuid4()),
                    model_id=request.model_id,
                    voice_id=request.voice_id,
                    text=text_chunk,
                    context=context,
                    params=request.tts_params.copy(),
                    alignment_type=request.alignment_type,
                    method="generate_request",
                    chunk_index=current_chunk_index,
                    created_at=request.created_at,
                    is_first=request.first_text_at is None,
                    start_time=request.start_time,
                    collected_time=request.collected_time,
                    text_span=text_span,
                    nonce=request.nonce,
                    target_sample_rate=target_sample_rate,
                    target_encoding=target_encoding,
                    first_audio_latency_started_at=request.first_text_at if current_chunk_index == 0 else None,
                    source_text=request.text_buffer,
                )
                to_send.append(item)
                if request.pending_chunks == 0:
                    request.start_time = monotonic()
                    request.collected_time = 0.0

                request.pending_chunks += 1
                request.commit_text(text_span[1] - request.text_committed_pos)
                log.debug(
                    "text_chunk_dispatched",
                    request_id=request.request_id,
                    chunk_index=current_chunk_index,
                    text_length=len(text_chunk),
                    pending_chunks=request.pending_chunks,
                    audio_queue_size=request.audio_queue.qsize(),
                    is_final=is_final,
                )

                pending_text = request.get_pending_text()
                if pending_text.strip():
                    request.generation_window_started_at = monotonic()
                    if not request.prepared_chunks:
                        self.schedule_timeout(request)
                else:
                    request.generation_window_started_at = None

        if to_send:
            log.info("engine_dispatch", request_count=len(to_send), active_streams=len(self._requests))
        self.executor.send_to_process(to_send)

    async def async_warmup(self, model_ids: list[str], voice_ids: list[str], num_warmup_tasks: int = 4):
        if len(model_ids) != len(voice_ids):
            raise ValueError("model_ids and voice_ids must have the same length")

        async def drain_stream(stream: TTSStream):
            async for _ in stream.pull_audio():
                pass

        for model_id, voice_id in zip(model_ids, voice_ids):
            stream = self.create_stream(model_id, voice_id=voice_id, params={})
            for i in range(5):
                stream.add_text("".join(["Hello, world!"] * (i + 1)))
                stream.force_generate()
                await drain_stream(stream)
            stream.close()

        for model_id, voice_id in zip(model_ids, voice_ids):
            streams = []
            for _ in range(num_warmup_tasks):
                stream = self.create_stream(model_id, voice_id=voice_id, params={})
                streams.append(stream)

            batch_size = num_warmup_tasks

            for i in range(batch_size):
                streams[i].add_text("Hello, world! Hello, world! Hello, world! Hello, world! Hello, world!")
                streams[i].force_generate()
            for stream in streams[:batch_size]:
                await drain_stream(stream)
            for stream in streams:
                stream.close()
            log.info("model_ready_to_respond", model_id=model_id, voice_id=voice_id)

    def warmup(self, model_ids: list[str], voice_ids: list[str], num_warmup_tasks: int = 4):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.async_warmup(model_ids, voice_ids, num_warmup_tasks))
        else:
            raise RuntimeError("warmup() cannot run inside an active event loop; use async_warmup() instead")

    def stop(self):
        with self._timeout_condition:
            self._stop_timeout = True
            self._timeout_condition.notify_all()
        if self.timeout_thread and self.timeout_thread.is_alive():
            self.timeout_thread.join(timeout=1.0)
        if self.executor:
            self.executor.stop()
