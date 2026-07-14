from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.engine import StreamingTTS


class AsyncStreamingTTS:
    def __init__(self, config: StreamingTTSConfig | StreamingTTS):
        self.engine = config if isinstance(config, StreamingTTS) else StreamingTTS(config)

    @classmethod
    def from_config(cls, config):
        return cls(StreamingTTS.from_config(config))

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def load_model(self, config):
        self.engine.load_model(config)

    def unload_model(self, model_id):
        self.engine.unload_model(model_id)

    def get_model_ids(self):
        return self.engine.get_model_ids()

    def get_model_infos(self):
        return self.engine.get_model_infos()

    def get_voice_ids(self, model_id):
        return self.engine.get_voice_ids(model_id)

    def create_stream(self, model_id, voice_id, params):
        return self.engine.create_stream(model_id, voice_id, params)

    async def generate(self, model_id: str, voice_id: str, text: str, params):
        stream = self.engine.create_stream(model_id, voice_id, params)
        stream.add_text(text)
        stream.force_generate()
        try:
            async for chunk in stream.pull_audio():
                yield chunk
        finally:
            stream.close()

    async def generate_full(self, model_id: str, voice_id: str, text: str, params):
        return await self.engine.generate_full(model_id, voice_id, text, params)

    async def async_warmup(self, model_ids, voice_ids, num_warmup_tasks=4):
        await self.engine.async_warmup(model_ids, voice_ids, num_warmup_tasks)

    def warmup(self, model_ids, voice_ids, num_warmup_tasks=4):
        self.engine.warmup(model_ids, voice_ids, num_warmup_tasks)

    def run(self):
        self.engine.run()

    def stop(self):
        self.engine.stop()
