import asyncio
import json

from tinfer.core.request import chunk_from_native


class TTSStream:
    def __init__(self, stream):
        self._stream = stream

    def add_text(self, text: str):
        self._stream.add_text(text)

    def force_generate(self):
        self._stream.force_generate()

    def try_generate(self):
        self._stream.try_generate()

    def cancel(self):
        self._stream.cancel()

    async def wait_for_audio(self):
        chunk = await asyncio.to_thread(self._stream.recv)
        return None if chunk is None else chunk_from_native(chunk)

    async def pull_audio(self):
        while (chunk := await self.wait_for_audio()) is not None:
            yield chunk

    def get_audio(self):
        return [chunk_from_native(chunk) for chunk in self._stream.get_audio()]

    def collect_audio(self):
        chunks = []
        while chunk := self._stream.recv():
            chunks.append(chunk_from_native(chunk))
        return chunks

    def get_state(self):
        return json.loads(self._stream.get_state())

    def close(self):
        self._stream.close()
