from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.web_ws import WebSocketResponse

from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.core.request import VoiceInfo
from tinfer.server.health import HealthState
from tinfer.support.observability import get_logger, record_span_exception, start_span
from .handler import WebSocketHandler
from .http_handler import SpeechHttpHandler
from .multi_context_handler import MultiContextHandler
from .model_resolver import ModelResolver
from .query_parser import parse_query, validate_language
from .schemas import Transport
from .response_formatter import format_model, format_voice

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
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._running = False
        model_infos = self.tts.get_model_infos()
        if not model_infos:
            raise ValueError("at least one model is required")
        self.models = ModelResolver(model_infos, model_infos[0].model_id)
        self._http_handler = SpeechHttpHandler(tts, self.health, self.models)

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/health/live", self._handle_live)
        app.router.add_get("/health/ready", self._handle_ready)
        app.router.add_get("/livez", self._handle_live)
        app.router.add_get("/readyz", self._handle_ready)
        app.router.add_get("/v1/models", self._handle_list_models)
        app.router.add_get("/v1/voices", self._handle_list_voices)
        app.router.add_post(
            "/v1/text-to-speech/{voice_id}/stream/with-timestamps",
            self._http_handler.stream_timing,
        )
        app.router.add_post(
            "/v1/text-to-speech/{voice_id}/with-timestamps",
            self._http_handler.unary_timing,
        )
        app.router.add_post(
            "/v1/text-to-speech/{voice_id}/stream",
            self._http_handler.stream_audio,
        )
        app.router.add_post(
            "/v1/text-to-speech/{voice_id}",
            self._http_handler.unary_audio,
        )
        app.router.add_get(
            "/v1/text-to-speech/{voice_id}/multi-stream-input",
            self._handle_multi_websocket,
        )
        app.router.add_get("/v1/text-to-speech/{voice_id}/stream-input", self._handle_websocket)
        return app

    async def start(self) -> None:
        if self._running:
            return
        self._shutdown_event = asyncio.Event()
        self._app = self.create_app()
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

    async def _handle_live(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"live": self.health.live, "status": self.health.status},
            status=200 if self.health.live else 503,
        )

    async def _handle_ready(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"ready": self.health.ready, "status": self.health.status},
            status=200 if self.health.ready else 503,
        )

    async def _handle_list_models(self, request: web.Request) -> web.Response:
        model_infos = self.tts.get_model_infos()
        models = [format_model(info) for info in model_infos]
        log.info("websocket_list_models", model_count=len(models))
        return web.json_response(models)

    async def _handle_list_voices(self, request: web.Request) -> web.Response:
        model_id = request.query["model_id"] if "model_id" in request.query else None
        model_ids = [model_id] if model_id else self.tts.get_model_ids()
        voices = []
        for mid in model_ids:
            try:
                voice_ids = self.tts.get_voice_ids(mid)
            except ValueError as error:
                return web.json_response({"error": str(error)}, status=404)
            voices.extend(
                format_voice(VoiceInfo(vid, vid, "generated", mid))
                for vid in voice_ids
            )
        log.info("websocket_list_voices", model_count=len(model_ids), voice_count=len(voices))
        return web.json_response({"voices": voices})

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
            error = self._websocket_request_error(request)
            if error is not None:
                return error
            await ws.prepare(request)
        
            voice_id = request.match_info["voice_id"]
            query_params = dict(request.query)
            peer = request.remote
            requested_model = query_params["model_id"] if "model_id" in query_params else None
            model_id = self.models.resolve(requested_model).model_id
            
            handler = WebSocketHandler(
                ws=ws,
                tts=self.tts,
                voice_id=voice_id,
                query_params=query_params,
                models=self.models,
            )
            
            with start_span(
                "websocket.connection",
                __name__,
                kind="server",
                attributes={
                    "network.peer.address": peer or "",
                    "tinfer.voice_id": voice_id,
                    "tinfer.model_id": model_id,
                },
            ) as span:
                log.info(
                    "websocket_connected",
                    peer=peer,
                    voice_id=voice_id,
                    model_id=model_id,
                )
                try:
                    await handler.handle()
                except Exception as error:
                    record_span_exception(span, error)
                    log.exception("websocket_handler_failed")
                    if not ws.closed:
                        await ws.close(code=1011, message=f"Server error: {str(error)}")
                finally:
                    await self._cleanup_handler(handler, "websocket_cleanup_failed")
                    log.info("websocket_disconnected")
        finally:
            await self.health.release_connection()
        
        return ws

    async def _handle_multi_websocket(self, request: web.Request) -> WebSocketResponse:
        if not await self.health.try_acquire_connection():
            return web.json_response({"error": "server is not accepting synthesis connections"}, status=503)
        ws = WebSocketResponse()
        handler = None
        try:
            error = self._websocket_request_error(request)
            if error is not None:
                return error
            await ws.prepare(request)
            handler = MultiContextHandler(
                ws=ws,
                tts=self.tts,
                voice_id=request.match_info["voice_id"],
                query_params=dict(request.query),
                models=self.models,
            )
            await handler.handle()
        except ValueError as error:
            if not ws.closed:
                await ws.send_json({"error": str(error)})
                await ws.close(code=1008)
        finally:
            try:
                if handler is not None:
                    await self._cleanup_handler(handler, "multi_websocket_cleanup_failed")
            finally:
                await self.health.release_connection()
        return ws

    async def _cleanup_handler(self, handler, failure_event: str) -> None:
        cleanup_task = asyncio.create_task(handler.cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise
        except Exception:
            log.exception(failure_event)

    def _websocket_request_error(self, request: web.Request) -> web.Response | None:
        try:
            query = parse_query(request, Transport.WEBSOCKET)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=422)
        try:
            model_info = self.models.resolve(query.model_id)
            validate_language(query, model_info)
        except ValueError as error:
            status = 404 if str(error).startswith("unknown model:") else 422
            return web.json_response({"error": str(error)}, status=status)
        voice_id = request.match_info["voice_id"]
        if voice_id not in self.tts.get_voice_ids(model_info.model_id):
            return web.json_response({"error": f"unknown voice: {voice_id}"}, status=404)
        return None
