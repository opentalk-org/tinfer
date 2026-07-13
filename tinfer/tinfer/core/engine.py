import asyncio
import threading
from typing import Any

import numpy as np
import yaml

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.request import (
    Alignment,
    AlignmentItem,
    AudioChunk,
    StreamParams,
    TTSRequest,
)
from tinfer.core.stream import TTSStream
from tinfer.core.scheduler import SchedulingMixin
from tinfer.executor.process_executor import ProcessExecutor
from tinfer.support.errors import InferenceError
from tinfer.support.observability import get_logger
from tinfer.utils.text_chunker import TextChunker

log = get_logger(__name__)


class StreamingTTS(SchedulingMixin):
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
            raise ValueError("Model must be loaded before registration. Call model.load(path) first.")
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

    def get_model_infos(self):
        return self.executor.get_model_infos()

    def get_voice_ids(self, model_id: str) -> list[str]:
        return self.executor.get_voice_ids(model_id)

    def run(self):
        self._stop_timeout = False
        self.timeout_thread = threading.Thread(target=self.run_timeout_loop, daemon=True)
        self.timeout_thread.start()

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
