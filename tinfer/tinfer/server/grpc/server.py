from __future__ import annotations
import grpc
from grpc.aio import Server

from .service import StyleTTSService
from . import styletts_pb2_grpc
from concurrent import futures

from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.server.health import HealthState
from tinfer.support.observability import get_logger

log = get_logger(__name__)


class GrpcHealthService:
    _SERVING_RESPONSE = b"\x08\x01"
    _NOT_SERVING_RESPONSE = b"\x08\x02"
    _LIVENESS_SERVICES = frozenset({"", "liveness", "live"})

    def __init__(self, health: HealthState) -> None:
        self.health = health

    @staticmethod
    def _parse_service(request: bytes) -> str:
        if not request or request[0] != 0x0A:
            return ""
        idx, length, shift = 1, 0, 0
        while idx < len(request):
            byte = request[idx]
            idx += 1
            length |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return request[idx:idx + length].decode("utf-8", "replace")

    def Check(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        service = self._parse_service(request)
        if service in self._LIVENESS_SERVICES:
            serving = self.health.live
        else:
            serving = self.health.ready
        return self._SERVING_RESPONSE if serving else self._NOT_SERVING_RESPONSE

    def register(self, server: grpc.aio.Server) -> None:
        rpc_method_handlers = {
            "Check": grpc.unary_unary_rpc_method_handler(
                self.Check,
                request_deserializer=lambda request: request,
                response_serializer=lambda response: response,
            ),
        }
        generic_handler = grpc.method_handlers_generic_handler("grpc.health.v1.Health", rpc_method_handlers)
        server.add_generic_rpc_handlers((generic_handler,))
        server.add_registered_method_handlers("grpc.health.v1.Health", rpc_method_handlers)


class GRPCServer:
    def __init__(self, tts: AsyncStreamingTTS, port: int = 50051, health: HealthState | None = None) -> None:
        self.tts = tts
        self.port = port
        self.health = health or HealthState(warmup_complete=True)
        self._server: grpc.aio.Server | None = None
        self._thread_pool: futures.ThreadPoolExecutor | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=10)
        self._server = grpc.aio.server(self._thread_pool)
        
        service = StyleTTSService(self.tts, health=self.health)
        health_service = GrpcHealthService(self.health)
        styletts_pb2_grpc.add_StyleTTSServiceServicer_to_server(service, self._server)
        health_service.register(self._server)
        
        listen_addr = f"[::]:{self.port}"
        self._server.add_insecure_port(listen_addr)
        
        await self._server.start()
        self._running = True
        log.info("grpc_server_started", port=self.port)

    async def stop(self, grace_period: float = 5.0) -> None:
        if not self._running or self._server is None:
            return
        
        await self._server.stop(grace_period)
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
            self._thread_pool = None
        self._running = False
        self._server = None
        log.info("grpc_server_stopped", port=self.port)

    async def serve(self) -> None:
        if not self._running:
            await self.start()
        
        if self._server is None:
            return
        
        await self._server.wait_for_termination()
