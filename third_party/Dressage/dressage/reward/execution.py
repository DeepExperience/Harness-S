"""Keep synchronous rewards off the rollout event loop.

The dedicated, process-local pool also bounds replay subprocess concurrency.
It is separate from asyncio's default executor used by trajectory log writers.
Native async rewards retain their existing event-loop execution semantics.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import inspect
import logging
import os
import threading
import time
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _sync_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            raw_limit = os.environ.get("DRESSAGE_REWARD_CONCURRENCY", "4")
            try:
                limit = int(raw_limit)
            except ValueError as exc:
                raise ValueError("DRESSAGE_REWARD_CONCURRENCY must be a positive integer") from exc
            if limit < 1:
                raise ValueError("DRESSAGE_REWARD_CONCURRENCY must be a positive integer")
            _EXECUTOR = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="dressage-reward")
            LOGGER.info("Synchronous reward executor started: concurrency=%d", limit)
        return _EXECUTOR


async def execute_reward(
    name: str,
    fn: Callable,
    sample: Any,
    *,
    args: Any = None,
    **kwargs: Any,
) -> float:
    """Run one reward without changing its value, order, or failure policy.

    Cancelling a queued reward cancels that job. A running Python thread cannot
    be killed safely: cancellation drains it asynchronously before propagating,
    so no late reward writes can race reuse of the Sample and its worker slot
    remains occupied until the real work ends. The reward's own subprocess/API
    timeout must still bound that work. Other rollouts keep making progress.
    """
    submitted = time.monotonic()
    timing: dict[str, Any] = {"name": name, "backend": "async", "outcome": "error"}
    worker_timing: dict[str, float] = {}
    metadata = getattr(sample, "metadata", None)
    session_id = metadata.get("session_id", "?") if isinstance(metadata, dict) else "?"

    def invoke_sync():
        started = time.monotonic()
        worker_timing["queue_seconds"] = started - submitted
        LOGGER.info("reward execution started name=%s session=%s queue_seconds=%.3f",
                    name, session_id, worker_timing["queue_seconds"])
        try:
            return fn(sample, args=args, **kwargs)
        finally:
            worker_timing["execution_seconds"] = time.monotonic() - started

    try:
        is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        )
        if is_async:
            timing["queue_seconds"] = 0.0
            try:
                result = await fn(sample, args=args, **kwargs)
            finally:
                timing["execution_seconds"] = time.monotonic() - submitted
        else:
            timing["backend"] = "thread"
            context = contextvars.copy_context()
            job = _sync_executor().submit(context.run, invoke_sync)
            future = asyncio.wrap_future(job)
            try:
                result = await asyncio.shield(future)
            except asyncio.CancelledError:
                timing["outcome"] = "cancelled"
                if not job.cancel():
                    # Shield repeatedly: a second cancellation must not release
                    # ownership while the running reward still mutates Sample.
                    while not future.done():
                        try:
                            await asyncio.shield(future)
                        except asyncio.CancelledError:
                            continue
                        except BaseException:
                            break
                    if future.done() and not future.cancelled():
                        try:
                            abandoned = future.result()
                            if inspect.iscoroutine(abandoned):
                                abandoned.close()
                        except BaseException:
                            pass  # Cancellation wins; consume the worker error.
                raise
            finally:
                timing.update(worker_timing)
            if inspect.isawaitable(result):
                # Backwards compatibility for sync factories of awaitables.
                started = time.monotonic()
                try:
                    result = await result
                finally:
                    timing["execution_seconds"] += time.monotonic() - started
        reward = float(result)
        timing["outcome"] = "completed"
        return reward
    except asyncio.CancelledError:
        timing["outcome"] = "cancelled"
        raise
    finally:
        timing["total_seconds"] = time.monotonic() - submitted
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict):
            metadata["reward_execution"] = dict(timing)
        LOGGER.info(
            "reward execution name=%s session=%s backend=%s outcome=%s "
            "queue_seconds=%.3f execution_seconds=%.3f total_seconds=%.3f",
            name, session_id,
            timing["backend"], timing["outcome"], timing.get("queue_seconds", 0.0),
            timing.get("execution_seconds", 0.0), timing["total_seconds"],
        )
