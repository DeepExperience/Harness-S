"""Finite wall-clock budgets for the shared SWE-Gym execution adapter."""
import asyncio
import os
import signal
import time
from contextlib import contextmanager


@contextmanager
def sync_stage(stage, timeout, timings):
    """Bound a stage without borrowing time from an enclosing case deadline.

    The existing case alarm remains the hard limit. Waiting for a verifier
    slot is deliberately outside any preparation/grading stage context.
    """
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    case_deadline = started + remaining if remaining else None
    stage_deadline = started + timeout if timeout > 0 else None
    deadlines = [v for v in (case_deadline, stage_deadline) if v is not None]

    def expired(signum, frame):
        if case_deadline is not None and (stage_deadline is None or case_deadline <= stage_deadline):
            if callable(previous):
                return previous(signum, frame)
        raise TimeoutError(f'SWE-Gym {stage} wall-clock timeout')

    if deadlines:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.000001, min(deadlines) - started))
    try:
        yield
    finally:
        timings[stage + '_seconds'] = time.monotonic() - started
        if deadlines:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if case_deadline is not None:
                signal.setitimer(signal.ITIMER_REAL, max(.000001, case_deadline - time.monotonic()), interval)


def seconds(stage):
    defaults = {'agent': 1800, 'prepare': 900, 'verifier': 1200, 'case': 3600, 'cleanup': 60}
    return float(os.environ.get('HAPT_SWEGYM_' + stage.upper() + '_TIMEOUT_SECONDS', defaults.get(stage, 0)))


async def bounded(operation, stage, *, timeout=None):
    timeout = seconds(stage) if timeout is None else timeout
    if timeout <= 0:
        return await operation
    task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done:
            return task.result()
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=seconds('cleanup') or 60)
        if not done:
            raise RuntimeError('SWEGYM_SERVICE_REBUILD_REQUIRED: cancellation did not finish')
        # Retrieve any cancellation/cleanup exception, but never treat a late
        # result as success after the deadline.
        if not task.cancelled():
            error = task.exception()
            if error:
                from dressage.rollout.case_failure import is_global_failure
                if is_global_failure(error):
                    raise error
        raise TimeoutError(f'SWE-Gym {stage} wall-clock timeout after {timeout:g}s')
    except asyncio.CancelledError:
        task.cancel()
        # External cancellation remains cancellation, not a zero reward.
        try:
            await asyncio.wait({task}, timeout=seconds('cleanup') or 60)
        finally:
            raise


async def cancel_session(client, session_id):
    if not session_id:
        return
    result = await asyncio.wait_for(client.cancel_session(session_id), timeout=seconds('cleanup') or 60)
    if not result.get('quiesced', False):
        raise RuntimeError('SWEGYM_SERVICE_REBUILD_REQUIRED: session requests did not stop')
