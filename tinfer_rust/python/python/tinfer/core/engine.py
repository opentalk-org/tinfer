import asyncio
import json
from typing import Any

import yaml

from tinfer._native import NativeEngine
from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.request import ModelInfo, StreamParams, chunk_from_native
from tinfer.core.stream import TTSStream


class StreamingTTS:
    def __init__(self, config: StreamingTTSConfig, auto_run: bool = True):
        self.config = config
        self._engine = NativeEngine(config.dumps())

    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            config = StreamingTTSConfig(**config)
        elif not isinstance(config, StreamingTTSConfig):
            config = StreamingTTSConfig.from_yaml(config)
        return cls(config)

    def create_stream(self, model_id: str, voice_id: str, params: StreamParams | dict[str, Any]):
        return TTSStream(self._engine.create_stream(model_id, voice_id, _params(params)))

    async def generate_full(self, model_id: str, voice_id: str, text: str, params):
        chunk = await asyncio.to_thread(
            self._engine.generate_full, model_id, voice_id, text, _params(params)
        )
        return chunk_from_native(chunk)

    async def generate_full_batch(self, model_id: str, voice_id: str, texts: list[str], params):
        return await asyncio.gather(*(
            self.generate_full(model_id, voice_id, text, item)
            for text, item in zip(texts, params, strict=True)
        ))

    def load_model(self, config: dict[str, Any]):
        self._engine.load_model(yaml.safe_dump(config, sort_keys=False))

    def unload_model(self, model_id: str):
        self._engine.unload_model(model_id)

    def get_model_ids(self) -> list[str]:
        return self._engine.get_model_ids()

    def get_model_infos(self):
        return [ModelInfo(model, tuple(languages), default) for model, languages, default in self._engine.get_model_infos()]

    def get_voice_ids(self, model_id: str) -> list[str]:
        return self._engine.get_voice_ids(model_id)

    async def async_warmup(self, model_ids: list[str], voice_ids: list[str], num_warmup_tasks: int = 4):
        await asyncio.gather(*(
            self.generate_full(model, voice, "Hello, world!", {})
            for model, voice in zip(model_ids, voice_ids, strict=True)
            for _ in range(num_warmup_tasks)
        ))

    def warmup(self, model_ids: list[str], voice_ids: list[str], num_warmup_tasks: int = 4):
        asyncio.run(self.async_warmup(model_ids, voice_ids, num_warmup_tasks))

    def run(self):
        return None

    def stop(self):
        self._engine.stop()


def _params(params) -> str:
    value = dict(params)
    alignment = value.get("alignment_type")
    if alignment is not None:
        value["alignment_type"] = alignment.value
    return json.dumps(value)
