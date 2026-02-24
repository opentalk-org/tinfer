class StreamingARTTSModel(TTSModel):
    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_parallel_chunks(self) -> bool:
        return False
