from __future__ import annotations
import asyncio
from aiohttp import web
from aiohttp.web_ws import WebSocketResponse
from typing import Optional

from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.server.health import HealthState
from tinfer.support.observability import get_logger, record_span_exception, start_span
from .handler import WebSocketHandler

log = get_logger(__name__)


class WebSocketServer:
    def __init__(
        self,
        tts: AsyncStreamingTTS,
        host: str = "0.0.0.0",
        port: int = 8000,
        health: HealthState | None = None,
    ) -> None:
        self.tts = tts
        self.host = host
        self.port = port
        self.health = health or HealthState(warmup_complete=True)
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
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/v1/text-to-speech/{voice_id}/stream-input", self._handle_websocket)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._running = True
        log.info("websocket_server_started", host=self.host, port=self.port)

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
        log.info("websocket_server_stopped", host=self.host, port=self.port)

    async def serve(self) -> None:
        if not self._running:
            await self.start()
        if self._shutdown_event is None:
            return
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

    async def _handle_health(self, request: web.Request) -> web.Response:
        ready = self.health.ready
        return web.json_response(
            {
                "ready": ready,
                "status": "serving" if ready else "not_serving",
            },
            status=200 if ready else 503,
        )

    async def _handle_websocket(self, request: web.Request) -> WebSocketResponse:
        if not await self.health.try_acquire_connection():
            return web.json_response(
                {
                    "error": "Server is not accepting new synthesis connections",
                    "status": self.health.status,
                },
                status=503,
            )

        ws = WebSocketResponse()
        try:
            await ws.prepare(request)
        
            voice_id = request.match_info.get("voice_id", "")
            query_params = dict(request.query)
            peer = request.remote
            
            handler = WebSocketHandler(
                ws=ws,
                tts=self.tts,
                voice_id=voice_id,
                query_params=query_params,
            )
            
            with start_span(
                "websocket.connection",
                __name__,
                kind="server",
                attributes={
                    "network.peer.address": peer or "",
                    "tinfer.voice_id": voice_id,
                    "tinfer.model_id": query_params.get("model_id", "styletts2"),
                },
            ) as span:
                log.info(
                    "websocket_connected",
                    peer=peer,
                    voice_id=voice_id,
                    model_id=query_params.get("model_id", "styletts2"),
                )
                try:
                    await handler.handle()
                except Exception as e:
                    record_span_exception(span, e)
                    log.exception("websocket_handler_failed")
                    if not ws.closed:
                        await ws.close(code=1011, message=f"Server error: {str(e)}")
                finally:
                    await handler.cleanup()
                    log.info("websocket_disconnected")
        finally:
            await self.health.release_connection()
        
        return ws
