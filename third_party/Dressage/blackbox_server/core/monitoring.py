from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class BackendMonitor:
    def __init__(
        self,
        interval_seconds: float,
        check: Callable[[], Awaitable[bool]],
        on_failure: Callable[[], Awaitable[None]],
    ) -> None:
        self._interval_seconds = interval_seconds
        self._check = check
        self._on_failure = on_failure
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval_seconds)
                return
            except asyncio.TimeoutError:
                pass

            healthy = await self._check()
            if not healthy:
                await self._on_failure()
                return
