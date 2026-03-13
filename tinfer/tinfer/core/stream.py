from tinfer.core.request import TTSRequest
from typing import Any
import asyncio
import queue

class TTSStream:
    def __init__(self, request: TTSRequest, engine):
        self._request = request
        self._engine = engine

    def add_text(self, text: str):
        self._request.append_text(text)
        self._signal_input()

    def _signal_input(self):
        self._engine.signal_input()

    def force_generate(self):
        self._request.force_next_generation = True
        self._signal_input()

    def cancel(self):
        self._engine._cancel_request(self._request.request_id)

    async def wait_for_audio(self):
        loop = asyncio.get_running_loop()
        chunk = await loop.run_in_executor(None, self._request.audio_queue.get)
        return chunk

    async def pull_audio(self):
        """Yield audio chunks. If a chunk has .error set, it indicates inference failure for this request."""
        has_text_to_process = len(self._request.get_pending_text()) > 0
        first_chunk_waited = False
        
        while True:
            has_pending = self._request.pending_chunks > 0
            queue_not_empty = not self._request.audio_queue.empty()
            
            if has_pending or queue_not_empty:
                chunk = await self.wait_for_audio()
                first_chunk_waited = True
                yield chunk
            elif has_text_to_process and not first_chunk_waited:
                chunk = await self.wait_for_audio()
                first_chunk_waited = True
                yield chunk
            else:
                break

    def get_audio(self):
        """Return list of audio chunks. Any chunk with .error set indicates inference failure for this request."""
        audio_chunks = []
        while True:
            if self._request.pending_chunks > 0 or not self._request.audio_queue.empty():
                chunk = self._request.audio_queue.get()
                audio_chunks.append(chunk)
            else:
                break

        return audio_chunks

    def get_state(self):
        return self._request.get_state()

    def close(self):
        self._request.audio_queue.queue.clear()
        self._engine._close_stream(self._request.request_id)