from typing import Any

from tinfer.core.request import TTSRequest, TTSRequestIPC, StreamParams
from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.stream import TTSStream
from tinfer.utils.text_chunker import TextChunker
import uuid
import yaml
import threading
from tinfer.executor.process_executor import ProcessExecutor
import queue
import time
from tinfer.core.request import AudioChunk, AudioChunkIPC
from time import monotonic
import numpy as np
from tinfer.core.request import Alignment, AlignmentItem
from tinfer.utils.audio_encoder import AudioFormat, parse_output_format
from tinfer.errors import InferenceError

_STREAM_PARAM_KEYS = frozenset({
    "chunk_length_schedule",
    "min_chunk_length_schedule",
    "min_chars_trigger",
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
        self.timeout_queue = queue.Queue()
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

    def generate_full(self, model_id: str, voice_id: str, text: str, params: StreamParams | dict[str, Any]):
        stream = self.create_stream(model_id, voice_id, params)

        stream.add_text(text)
        stream.force_generate()

        audio_chunks = stream.get_audio()

        stream.close()

        return self._merge_audio_chunks(audio_chunks)

    def generate_full_batch(self, model_id: str, voice_id: str, texts: list[str], params: list[StreamParams | dict[str, Any]]):
        streams: list[TTSStream] = []
        for text, params_dict in zip(texts, params):
            stream = self.create_stream(model_id, voice_id, params_dict)
            stream.add_text(text)
            stream.force_generate()
            streams.append(stream)

        results: list[AudioChunk] = []
        for stream in streams:
            audio_chunks = stream.get_audio()
            results.append(self._merge_audio_chunks(audio_chunks))

        for stream in streams:
            stream.close()

        return results

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

        char_offset = 0
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
                                char_start=it.char_start + char_offset,
                                char_end=it.char_end + char_offset,
                                start_ms=it.start_ms + time_offset_ms,
                                end_ms=it.end_ms + time_offset_ms,
                            )
                        )

            span_len = max(0, int(c.text_span[1]) - int(c.text_span[0]))
            char_offset += span_len

            num_samples = int(c.audio.shape[-1]) if hasattr(c.audio, "shape") and len(c.audio.shape) > 0 else int(len(c.audio))
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
            text_span=(0, char_offset),
            alignments=merged_alignment,
        )

    def _close_stream(self, request_id: str):
        if request_id not in self._requests:
            return

        self._cancel_request(request_id)
        self._requests.pop(request_id)

    def load_model(self, model_id: str, model_path: str, voices_folder: str | None = None):
        self.executor.load_model(model_id, model_path, voices_folder=voices_folder, compile_models=self.config.compile_models)

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

    def run(self):
        self._stop_timeout = False
        self.timeout_thread = threading.Thread(target=self.run_timeout_loop, daemon=True)
        self.timeout_thread.start()

    def run_timeout_loop(self):
        while not self._stop_timeout:
            t, request_id = self.timeout_queue.get()
            now = time.monotonic()

            if request_id in self._requests:
                request = self._requests[request_id]
                timeout_trigger_ms = request.timeout_trigger_ms
                wait_time = max(0.0, t + timeout_trigger_ms / 1000.0 - now)
                if wait_time > 0:
                    time.sleep(wait_time)
                
                self.signal_input()

    def process_results(self, results: list[AudioChunkIPC]):
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
            request.audio_queue.put(chunk)
            request.pending_chunks -= 1
            request.collected_time += len(result.audio) / result.sample_rate

        self.signal_input()

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
            "min_chunk_length_schedule": self.config.default_min_chunk_schedule.copy(),
            "min_chars_trigger": self.config.min_chars_trigger,
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
                if key in ("chunk_length_schedule", "min_chunk_length_schedule") and isinstance(val, list):
                    val = list(val)
                request_kwargs[key] = val

        request = TTSRequest(**request_kwargs)
        self._requests[request_id] = request

        stream = TTSStream(request, self)

        return stream

    def _cancel_request(self, request_id: str):
        if request_id not in self._requests:
            return

        self.executor.cancel_request(request_id)

        request = self._requests[request_id]
        request.text_buffer = ""
        request.text_committed_pos = 0
        request.force_next_generation = False
        request.pending_chunks = 0
        request.nonce = str(uuid.uuid4())
        request.audio_queue.queue.clear()

    def signal_input(self):
        now = monotonic()
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
                )
                to_send.append(item)
                if request.pending_chunks == 0:
                    request.start_time = monotonic()
                    request.collected_time = 0.0

                request.pending_chunks += 1
                request.commit_text(len(text_chunk))

                if not is_final and len(request.get_pending_text()) > 0:
                    self.timeout_queue.put((now, request.request_id))

        self.executor.send_to_process(to_send)

    def warmup(self, model_ids: list[str], voice_ids: list[str], num_warmup_tasks: int = 4):
        if len(model_ids) != len(voice_ids):
            raise ValueError("model_ids and voice_ids must have the same length")

        for model_id, voice_id in zip(model_ids, voice_ids):
            stream = self.create_stream(model_id, voice_id=voice_id, params={})
            for i in range(5):
                stream.add_text("".join(["Hello, world!"] * (i + 1)))
                stream.force_generate()
                audio_chunks = stream.get_audio()
                del audio_chunks
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
                audio_chunks = stream.get_audio()
                del audio_chunks
            for stream in streams:
                stream.close()

    def stop(self):
        self._stop_timeout = True
        if self.timeout_thread and self.timeout_thread.is_alive():
            self.timeout_thread.join(timeout=1.0)
        if self.executor:
            self.executor.stop()



