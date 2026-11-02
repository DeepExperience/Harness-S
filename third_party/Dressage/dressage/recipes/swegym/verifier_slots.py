"""Process-safe verifier slots shared by rollout workers and held-out evaluators.

Locks are released by the OS on process exit. Acquiring asynchronously never
blocks the event loop and cancellation cannot strand a background lock owner.
"""
import asyncio
from contextlib import asynccontextmanager, contextmanager
import fcntl
from functools import wraps
import os
from pathlib import Path
import time


def _try_acquire():
    count = int(os.environ.get("DRESSAGE_SWEGYM_VERIFIER_CONCURRENCY", "32"))
    if count < 1:
        raise ValueError("verifier concurrency must be positive")
    root = Path(os.environ.get("DRESSAGE_SWEGYM_VERIFIER_SLOT_DIR", "/tmp/dressage-swegym-verifier-slots"))
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        handle = (root / f"{index}.lock").open("a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
        else:
            return handle
    return None


@contextmanager
def verifier_slot():
    handle = None
    try:
        while handle is None:
            handle = _try_acquire()
            if handle is None:
                time.sleep(.25)
        yield
    finally:
        if handle is not None:
            handle.close()


@asynccontextmanager
async def async_verifier_slot():
    handle = None
    try:
        while handle is None:
            handle = _try_acquire()
            if handle is None:
                await asyncio.sleep(.25)
        yield
    finally:
        if handle is not None:
            handle.close()


def limited_verifier(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        metadata = kwargs.get("metadata", args[2] if len(args) > 2 else {})
        timings = metadata.setdefault("swegym_eval_timings", {})
        started = time.monotonic()
        try:
            async with async_verifier_slot():
                timings["queue_seconds"] = time.monotonic() - started
                return await function(*args, **kwargs)
        finally:
            timings.setdefault("queue_seconds", time.monotonic() - started)
    return wrapped
