from __future__ import annotations
import grpc
from grpc.aio import Server
from .service import StyleTTSService
from . import styletts_pb2_grpc
from . import styletts_pb2
import asyncio
from concurrent import futures

from tinfer.core.async_engine import AsyncStreamingTTS

class GRPCServer:
    def __init__(self, tts: AsyncStreamingTTS, port: int = 50051) -> None:
        self.tts = tts
        self.port = port
        self._server: grpc.aio.Server | None = None
        self._thread_pool: futures.ThreadPoolExecutor | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=10)
        self._server = grpc.aio.server(self._thread_pool)
        
        service = StyleTTSService(self.tts)
        styletts_pb2_grpc.add_StyleTTSServiceServicer_to_server(service, self._server)
        
        listen_addr = f"[::]:{self.port}"
        self._server.add_insecure_port(listen_addr)
        
        await self._server.start()
        self._running = True

    async def stop(self, grace_period: float = 5.0) -> None:
        if not self._running or self._server is None:
            return
        
        await self._server.stop(grace_period)
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
            self._thread_pool = None
        self._running = False
        self._server = None

    async def serve(self) -> None:
        if not self._running:
            await self.start()
        
        if self._server is None:
            return
        
        await self._server.wait_for_termination()
