from __future__ import annotations
import asyncio
from aiohttp import web
from aiohttp.web_ws import WebSocketResponse
from typing import Optional

from tinfer.core.async_engine import AsyncStreamingTTS
from .handler import WebSocketHandler


class WebSocketServer:
    def __init__(self, tts: AsyncStreamingTTS, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.tts = tts
        self.host = host
        self.port = port
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._shutdown_event = asyncio.Event()
        self._app = web.Application()
        self._app.router.add_get("/v1/text-to-speech/{voice_id}/stream-input", self._handle_websocket)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._running = True

    async def stop(self, grace_period: float = 5.0) -> None:
        if not self._running or self._shutdown_event is None:
            return
        self._shutdown_event.set()
        await asyncio.sleep(grace_period)
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._running = False
        self._app = None
        self._runner = None
        self._site = None
        self._shutdown_event = None

    async def serve(self) -> None:
        if not self._running:
            await self.start()
        if self._shutdown_event is None:
            return
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

    async def _handle_websocket(self, request: web.Request) -> WebSocketResponse:
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        voice_id = request.match_info.get("voice_id", "")
        query_params = dict(request.query)
        
        handler = WebSocketHandler(
            ws=ws,
            tts=self.tts,
            voice_id=voice_id,
            query_params=query_params,
        )
        
        try:
            await handler.handle()
        except Exception as e:
            if not ws.closed:
                await ws.close(code=1011, message=f"Server error: {str(e)}")
        finally:
            await handler.cleanup()
        
        return ws
