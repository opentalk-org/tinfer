from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from tinfer.server.websocket.response_formatter import format_ws_audio
from tinfer.server.websocket.schemas import SpeechOutputFormat

SendResponse = Callable[[dict[str, Any]], Awaitable[None]]
CloseResponse = Callable[[], Awaitable[None]]


class StreamContext:
    def __init__(
        self,
        stream,
        output_format: SpeechOutputFormat,
        send_response: SendResponse,
        inactivity_timeout: float,
        close_response: CloseResponse,
        context_id: str | None = None,
    ) -> None:
        self.stream = stream
        self.output_format = output_format
        self.send_response = send_response
        self.inactivity_timeout = inactivity_timeout
        self.close_response = close_response
        self.context_id = context_id
        self.closed = False
        self._close_started = False
        self._generation_queue: asyncio.Queue[None] = asyncio.Queue()
        self._pump_task = asyncio.create_task(self._pump_audio())
        self._inactivity_task = asyncio.create_task(self._watch_inactivity())

    async def add_text(self, text: str) -> None:
        if self._close_started:
            raise ValueError("stream context is finalizing")
        self._reset_inactivity()
        self.stream.add_text(text)
        self._generation_queue.put_nowait(None)

    async def try_generate(self) -> None:
        if self._close_started:
            raise ValueError("stream context is finalizing")
        self._reset_inactivity()
        self.stream.try_generate()
        self._generation_queue.put_nowait(None)

    async def flush(self) -> None:
        if self._close_started:
            raise ValueError("stream context is finalizing")
        self._reset_inactivity()
        self.stream.force_generate()
        self._generation_queue.put_nowait(None)

    async def finalize(self) -> None:
        if self._close_started:
            return
        self._close_started = True
        self._cancel_inactivity()
        self.stream.force_generate()
        self._generation_queue.put_nowait(None)
        try:
            await self._drain_generation()
            final: dict[str, Any] = {"isFinal": True}
            if self.context_id is not None:
                final["contextId"] = self.context_id
            await self.send_response(final)
        finally:
            await self.close()

    async def keepalive(self) -> None:
        if self._close_started:
            raise ValueError("stream context is finalizing")
        self._reset_inactivity()

    async def close(self) -> None:
        if self.closed:
            return
        self.abort()
        current = asyncio.current_task()
        if self._pump_task is not current:
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._cancel_inactivity()
        if not self._pump_task.done():
            self._pump_task.cancel()
        self.stream.close()

    async def _pump_audio(self) -> None:
        try:
            while not self.closed:
                await self._generation_queue.get()
                try:
                    async for chunk in self.stream.pull_audio():
                        if chunk.error is not None:
                            raise RuntimeError(chunk.error)
                        await self.send_response(
                            format_ws_audio(
                                chunk,
                                self.output_format,
                                is_final=False,
                                context_id=self.context_id,
                            )
                        )
                        self._reset_inactivity()
                finally:
                    self._generation_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def _drain_generation(self) -> None:
        queue_drained = asyncio.create_task(self._generation_queue.join())
        done, _ = await asyncio.wait(
            (queue_drained, self._pump_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._pump_task in done:
            queue_drained.cancel()
            try:
                await queue_drained
            except asyncio.CancelledError:
                pass
            await self._pump_task
            raise RuntimeError("audio pump stopped before generation drained")
        await queue_drained

    async def _watch_inactivity(self) -> None:
        try:
            await asyncio.sleep(self.inactivity_timeout)
            if not self.closed:
                error: dict[str, Any] = {"error": "inactivity timeout"}
                if self.context_id is not None:
                    error["contextId"] = self.context_id
                await self.send_response(error)
                await self.close_response()
        except asyncio.CancelledError:
            pass

    def _reset_inactivity(self) -> None:
        self._cancel_inactivity()
        self._inactivity_task = asyncio.create_task(self._watch_inactivity())

    def _cancel_inactivity(self) -> None:
        if not self._inactivity_task.done():
            self._inactivity_task.cancel()
