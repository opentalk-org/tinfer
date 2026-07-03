from __future__ import annotations
import grpc
from grpc.aio import Server
from grpc_health.v1 import health_pb2, health_pb2_grpc

from .service import StyleTTSService
from . import styletts_pb2_grpc
from concurrent import futures

from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.server.health import HealthState
from tinfer.support.observability import get_logger

log = get_logger(__name__)


class GrpcHealthService(health_pb2_grpc.HealthServicer):
    def __init__(self, health: HealthState) -> None:
        self.health = health

    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        status = health_pb2.HealthCheckResponse.SERVING
        if not self.health.ready:
            status = health_pb2.HealthCheckResponse.NOT_SERVING
        return health_pb2.HealthCheckResponse(status=status)


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
        health_pb2_grpc.add_HealthServicer_to_server(health_service, self._server)
        
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
