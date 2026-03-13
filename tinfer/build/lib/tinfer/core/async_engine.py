from typing import Any

from tinfer.core.engine import StreamingTTS
from tinfer.core.request import StreamParams
from tinfer.config.engine_config import StreamingTTSConfig
import yaml


class AsyncStreamingTTS:
    def __init__(self, config: StreamingTTSConfig | StreamingTTS):
        if isinstance(config, StreamingTTS):
            self.engine = config
        else:
            self.engine = StreamingTTS(config)

    def from_config(cls, config: StreamingTTSConfig):
        if isinstance(config, dict):
            config = StreamingTTSConfig(**config)
        else:
            with open(config) as f:
                config_dict = yaml.safe_load(f)
            config = StreamingTTSConfig(**config_dict)
        return cls(config)

    def load_model(self, model_id: str, model_path: str, voices_folder: str | None = None):
        self.engine.load_model(model_id, model_path, voices_folder=voices_folder)

    def register_model(self, model_id: str, model, device: str | None = None, keep_in_main: bool = True):
        self.engine.register_model(model_id, model, device, keep_in_main)

    def unload_model(self, model_id: str):
        self.engine.unload_model(model_id)

    def create_stream(self, model_id: str, voice_id: str, params: StreamParams | dict[str, Any]):
        return self.engine.create_stream(model_id, voice_id, params)

    async def generate(self, model_id: str, voice_id: str, text: str, params: StreamParams | dict[str, Any]):
        stream = self.create_stream(model_id, voice_id, params)
        stream.add_text(text)
        stream.force_generate()
        try:
            async for chunk in stream.pull_audio():
                yield chunk
        finally:
            stream.close()

    def generate_full(self, model_id: str, voice_id: str, text: str, params: StreamParams | dict[str, Any]):
        return self.engine.generate_full(model_id, voice_id, text, params)

    def warmup(self, model_ids: list[str] = None, num_warmup_tasks: int = 4):
        self.engine.warmup(model_ids, num_warmup_tasks)

    def stop(self):
        self.engine.stop()