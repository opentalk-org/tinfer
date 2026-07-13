from __future__ import annotations

import heapq
import uuid
from time import monotonic
from typing import Any

from tinfer.core.request import AudioChunk, AudioChunkIPC, StreamParams, TTSRequest, TTSRequestIPC
from tinfer.core.stream import TTSStream
from tinfer.support.observability import get_logger
from tinfer.utils.audio_encoder import AudioFormat, parse_output_format

log = get_logger(__name__)

_STREAM_PARAM_KEYS = frozenset({
    "chunk_length_schedule",
    "timeout_trigger_ms",
    "alignment_type",
    "target_sample_rate",
    "target_encoding",
    "tts_params",
})


class SchedulingMixin:
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

