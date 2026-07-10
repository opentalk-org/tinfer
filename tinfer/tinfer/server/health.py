from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class HealthState:
    warmup_complete: bool = False
    draining: bool = False
    stopped: bool = False
    _active_connections: int = 0
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def ready(self) -> bool:
        return self.warmup_complete and not self.draining and not self.stopped

    @property
    def live(self) -> bool:
        return not self.stopped

    @property
    def accepting_connections(self) -> bool:
        return self.ready

    @property
    def active_connections(self) -> int:
        return self._active_connections

    @property
    def status(self) -> str:
        if self.stopped:
            return "stopped"
        if self.draining:
            return "draining"
        if self.warmup_complete:
            return "serving"
        return "starting"

    def mark_warmup_complete(self) -> None:
        self.warmup_complete = True

    async def try_acquire_connection(self) -> bool:
        async with self._condition:
            if not self.accepting_connections:
                return False
            self._active_connections += 1
            return True

    async def release_connection(self) -> None:
        async with self._condition:
            self._active_connections -= 1
            if self._active_connections <= 0:
                self._condition.notify_all()

    async def begin_draining(self) -> None:
        async with self._condition:
            self.draining = True
            self._condition.notify_all()

    async def mark_stopped(self) -> None:
        async with self._condition:
            self.stopped = True
            self._condition.notify_all()

    async def wait_for_no_active_connections(self) -> None:
        async with self._condition:
            while self._active_connections > 0:
                await self._condition.wait()
