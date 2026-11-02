"""Slime custom generation hook for blackbox sandbox rollouts."""

from __future__ import annotations

import logging
import asyncio
import os
import time
from typing import Any

from dressage.config import proxy_url
from dressage.paddock.blackbox.execute_hooks import (
    execute_blackbox_cmds_for_stage,
    parse_blackbox_execute_cmds,
)
from dressage.paddock.blackbox.failures import (
    HARVESTABLE_AGENT_ERRORS,
    agent_response_text,
    expected_abort_from_call_agent_exception,
    failure_from_call_agent_exception,
    failure_from_payload_state,
    record_blackbox_abort_for_retry,
    record_agent_early_stop_metadata,
    record_agent_failure_metadata,
)
from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    merge_backend_options,
    normalize_blackbox_type,
)
from dressage.paddock.lifecycle import (
    exception_summary as _exception_summary,
    schedule_terminate_paddock,
    terminate_paddock_best_effort,
)
from dressage.rollout import multi_segment
from dressage.rollout.artifacts.samples import (
    instance_id as _instance_id,
    set_status as _set_status,
)
from dressage.rollout.artifacts.writer import DEFAULT_WRITER as _ARTIFACT_WRITER
from dressage.rollout.prewarm import (
    claim_prewarm,
    ensure_blackbox_session_id,
)
from dressage.rollout.generate.runtime import (
    get_paddock_from_env,
    get_proxy_client,
    maybe_await,
    paddock_env_args_from_metadata,
)

logger = logging.getLogger(__name__)


def _await_paddock_termination() -> bool:
    """Return whether reuse depends on completing teardown in this task.

    With a fixed-size local Docker pool, the next trajectory cannot acquire a
    slot until the previous release/reset completes.  Running that reset as a
    background lifecycle task lets CPU-heavy reward/token post-processing
    starve it on the rollout event loop, producing synchronized 30/60-second
    release timeouts.  Fresh-container providers retain background teardown.
    """

    if os.environ.get("DRESSAGE_SANDBOX_PROVIDER") != "docker":
        return False
    try:
        return int(os.environ.get("DRESSAGE_DOCKER_POOL_SIZE", "0")) > 0
    except ValueError:
        return False


def _chat_messages_from_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    return [{"role": "user", "content": str(prompt)}]


def _backend_options_for_register(
    *,
    args: Any,
    metadata: dict[str, Any],
    blackbox_type: str,
) -> Any:
    backend_options = metadata.get("backend_options")
    return merge_backend_options(blackbox_type, backend_options, args=args)


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    *, finalize_hook=None, strict_errors=False,
) -> Any:
    """Run one blackbox sandbox rollout and write proxy data back to Sample."""
    del sampling_params
    if evaluation and finalize_hook is None:
        raise ValueError("blackbox_dispatch does not support evaluation mode")

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata.pop("blackbox_error", None)
    metadata.pop("blackbox_error_log_path", None)
    metadata.pop("blackbox_expected_abort", None)
    metadata.pop("blackbox_agent_early_stop", None)
    metadata.pop("blackbox_agent_early_stop_kind", None)
    for key in ("blackbox_agent_error_kind", "blackbox_agent_state", "blackbox_agent_http_status_code",
                "blackbox_agent_error_details", "blackbox_agent_response", "blackbox_agent_response_truncated"):
        metadata.pop(key, None)
    metadata["execute_cmds"] = []
    session_id = ensure_blackbox_session_id(sample)
    instance_id = _instance_id(sample)
    metadata["session_id"] = session_id
    metadata["instance_id"] = instance_id
    blackbox_type = normalize_blackbox_type(
        metadata.get("blackbox_type") or DEFAULT_BLACKBOX_TYPE
    )

    extra_env_args = None
    if "blackbox_type" in metadata or blackbox_type != DEFAULT_BLACKBOX_TYPE:
        extra_env_args = {"blackbox_type": blackbox_type}
    env_args = paddock_env_args_from_metadata(
        metadata,
        extra_env_args=extra_env_args,
    )
    paddock = None
    proxy_client = None
    state = None
    initialized = False
    agent_response = ""
    trajectory_payload = None
    failure_phase = 'environment_initialization'
    try:
        execute_cmd_schedule = parse_blackbox_execute_cmds(
            metadata.get("blackbox_execute_cmds")
        )
        backend_options = _backend_options_for_register(
            args=args,
            metadata=metadata,
            blackbox_type=blackbox_type,
        )
        proxy_client = get_proxy_client()
        prewarm_requested = bool(metadata.get("prewarm_requested"))
        prewarm_t0 = time.monotonic()
        handle = await claim_prewarm(session_id)
        prewarm_wait = time.monotonic() - prewarm_t0
        if handle is not None:
            paddock = handle.paddock
            state = handle.state
            env_args = handle.env_args
            initialized = True
            # Only trajectories that requested a prewarm contribute to the
            # prewarm hit-rate metrics (see compute_prewarm_metrics).
            if prewarm_requested:
                metadata["prewarm_hit"] = 1
                metadata["prewarm_wait_seconds"] = prewarm_wait
            logger.debug(
                "claimed prewarmed sandbox for session_id=%s group_id=%d: "
                "wait=%.2fs",
                session_id,
                handle.group_id,
                prewarm_wait,
            )
        else:
            if prewarm_requested:
                metadata["prewarm_hit"] = 0
            previous_session_id = session_id
            sample.session_id = None
            session_id = ensure_blackbox_session_id(sample)
            metadata["session_id"] = session_id
            logger.debug(
                "initializing sandbox with a new session_id after prewarm miss: "
                "previous_session_id=%s new_session_id=%s",
                previous_session_id,
                session_id,
            )
            paddock = get_paddock_from_env(allow_whitebox_mode=False)
            state = await maybe_await(
                paddock.init(
                    session_id,
                    metadata.get("env_type"),
                    env_args,
                )
            )
            initialized = True
        if metadata.get("before_agent_files"):
            await maybe_await(paddock.write_files(state, files=metadata["before_agent_files"]))
        if not hasattr(paddock, "register_agent"):
            raise TypeError(f"{type(paddock).__name__} does not implement register_agent")
        registration = await maybe_await(
            paddock.register_agent(
                state,
                instance_id=instance_id,
                session_id=session_id,
                router_url=proxy_url(),
                blackbox_type=blackbox_type,
                backend_options=backend_options,
            )
        )
        if isinstance(registration, dict) and registration.get('agent_registered') is False:
            if (not evaluation or finalize_hook is None or
                    metadata.get('task_source') != 'dbbench' or
                    registration.get('termination') != 'benchmark_task_error'):
                raise RuntimeError('Unregistered agent cannot produce a training trajectory')
            # An official source initialization error has no model request and
            # therefore no proxy session. The trusted domain hook verifies its
            # evidence before accepting the official zero in the full denominator.
            await finalize_hook(metadata, {'data': []}, session_id=session_id)
            return {'status': 'completed', 'metadata': metadata, 'segments': 0}
        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="before_agent",
        )
        call_payload: Any = None
        failure_phase = 'rollout'
        call_succeeded = False
        try:
            call_payload = await maybe_await(
                paddock.call_agent(
                    state,
                    session_id=session_id,
                    messages=_chat_messages_from_prompt(sample.prompt),
                    metadata={"source": "dressage"} if finalize_hook else {"source": "dressage", **metadata},
                )
            )
            call_succeeded = True
        except Exception as exc:
            if agent_failure := failure_from_call_agent_exception(exc):
                record_agent_failure_metadata(metadata, agent_failure)
                if agent_failure.kind in HARVESTABLE_AGENT_ERRORS:
                    record_agent_early_stop_metadata(metadata, agent_failure)
                    logger.info(
                        "harvesting blackbox rollout after agent early stop: "
                        "session_id=%s kind=%s",
                        session_id,
                        agent_failure.kind,
                    )
                else:
                    raise agent_failure from exc
            else:
                raise

        if call_succeeded:
            agent_response = agent_response_text(call_payload)
            if agent_failure := failure_from_payload_state(
                call_payload,
                agent_response=agent_response,
            ):
                record_agent_failure_metadata(metadata, agent_failure)
                raise agent_failure

        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="after_agent",
        )
        failure_phase = 'trajectory_finalize'
        await proxy_client.finalize_session(
            session_id, instance_id=instance_id, label=getattr(sample, "label", None)
        )
        trajectory_payload = await proxy_client.read_trajectory(
            trajectory_id=session_id,
            instance_id=instance_id,
            # Read first, then idempotently discard after local artifacts and
            # Samples are durable.  A destructive read cannot distinguish a
            # lost response from a request that never reached the proxy.
            drain=False,
        )
        try:
            await _ARTIFACT_WRITER.write_session_payload(
                trajectory_payload,
                session_id=session_id,
                instance_id=instance_id,
            )
        except Exception:
            logger.warning(
                "failed to write trajectory payload log for session_id=%s",
                session_id,
                exc_info=True,
            )
        segments = trajectory_payload.get("data") or []
        if not segments and metadata.get("blackbox_agent_early_stop") is True:
            raise RuntimeError(
                "blackbox agent stopped early before the proxy recorded any "
                f"trainable segments: kind={metadata.get('blackbox_agent_early_stop_kind')}"
            )
        if finalize_hook is not None:
            failure_phase = 'grading_and_artifacts'
            await finalize_hook(metadata, trajectory_payload, session_id=session_id)
        if evaluation:
            await proxy_client.discard_session(session_id)
            return {"status": "completed", "metadata": metadata, "segments": len(segments)}
        if not segments:
            raise RuntimeError('Trajectory completed without generated tokens')
        failure_phase = 'token_conversion'
        base_metadata_for_logs = dict(metadata)
        result = multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=args,
            agent_response=agent_response,
            session_id=session_id,
            instance_id=instance_id,
        )
        log_template = sample
        try:
            await _ARTIFACT_WRITER.write_segment_samples(
                log_template,
                args=args,
                segments=segments,
                base_metadata=base_metadata_for_logs,
                session_id=session_id,
                instance_id=instance_id,
                agent_response=agent_response,
            )
        except Exception:
            logger.warning(
                "failed to write sample logs for session_id=%s",
                session_id,
                exc_info=True,
            )
        try:
            await proxy_client.discard_session(session_id)
        except Exception:
            # The retained finalized trajectory is safe and will be reclaimed
            # by proxy retention; never discard a valid Sample only because
            # best-effort in-memory cleanup failed.
            logger.warning(
                "failed to discard finalized proxy session=%s",
                session_id,
                exc_info=True,
            )
        return result
    except Exception as exc:
        from dressage.rollout.case_failure import (
            zero_case_failures, is_global_failure, failure_diagnostics,
            record_case_failure, zero_failed_samples,
        )
        if zero_case_failures() and expected_abort_from_call_agent_exception(exc) is None:
            if is_global_failure(exc, contract_errors=False):
                raise
            logger.warning('case failed; retaining reward zero: session=%s phase=%s error=%s',
                           session_id, failure_phase, _exception_summary(exc))
            diagnostics = failure_diagnostics(exc, phase=failure_phase, session_id=session_id)
            record_case_failure(sample, diagnostics)
            # Read exact proxy tokens before discarding. Never rerun a model
            # request or invent a logprob to replace an unsuccessful attempt.
            if trajectory_payload is None and initialized:
                try:
                    await asyncio.wait_for(proxy_client.finalize_session(
                        session_id, instance_id=instance_id, label=getattr(sample, 'label', None)), 60)
                except Exception as harvest_error:
                    if is_global_failure(harvest_error, contract_errors=False):
                        raise
                try:
                    trajectory_payload = await asyncio.wait_for(proxy_client.read_trajectory(
                        trajectory_id=session_id, instance_id=instance_id, drain=False), 60)
                except Exception as harvest_error:
                    if is_global_failure(harvest_error, contract_errors=False):
                        raise
                    diagnostics['harvest_error'] = _exception_summary(harvest_error)
            from dressage.rollout.generate.runtime import discard_proxy_session_best_effort
            await discard_proxy_session_best_effort(session_id, proxy_client=proxy_client)
            if evaluation:
                return {'status': 'case_failure', 'metadata': metadata,
                        'case_failure_diagnostics': diagnostics, 'segments': 0}
            retained = sample
            if trajectory_payload and trajectory_payload.get('data'):
                try:
                    retained = multi_segment.expand_segments_to_samples(
                        sample, trajectory_payload['data'], args=args, agent_response=agent_response,
                        session_id=session_id, instance_id=instance_id)
                except Exception as conversion_error:
                    if is_global_failure(conversion_error, contract_errors=False):
                        raise
                    diagnostics['conversion_error'] = _exception_summary(conversion_error)
            return zero_failed_samples(retained, exc, phase=failure_phase)
        if strict_errors:
            raise
        expected_abort = expected_abort_from_call_agent_exception(exc)
        if expected_abort is None:
            # Preserve the failed attempt's proxy transcript before container
            # cleanup. It is diagnostic only: never convert it into gradients.
            if initialized and os.environ.get("HAPT_TASK_KIND") == "alfworld" and os.environ.get("HAPT_JOINT_MODE") == "joint":
                try:
                    failed_payload = await proxy_client.read_trajectory(
                        trajectory_id=session_id, instance_id=instance_id, drain=False)
                    await _ARTIFACT_WRITER.write_session_payload(
                        {**failed_payload, "diagnostic_only": True},
                        session_id=session_id, instance_id=instance_id)
                    await _ARTIFACT_WRITER.drain()
                except Exception:
                    logger.warning("failed to preserve aborted session %s", session_id, exc_info=True)
            logger.warning(
                "blackbox rollout failed for session_id=%s: %s",
                session_id,
                _exception_summary(exc),
            )
            try:
                error_log_path = await _ARTIFACT_WRITER.write_error(
                    exc,
                    sample=sample,
                    metadata=dict(metadata),
                    session_id=session_id,
                    instance_id=instance_id,
                    blackbox_type=blackbox_type,
                    env_args=dict(env_args),
                    state=state,
                    agent_response=agent_response,
                )
                if error_log_path is not None:
                    metadata["blackbox_error_log_path"] = str(error_log_path)
            except Exception:
                logger.warning(
                    "failed to write trajectory error log for session_id=%s",
                    session_id,
                    exc_info=True,
                )
            record_blackbox_abort_for_retry(metadata, session_id, exc)
        else:
            metadata["blackbox_expected_abort"] = expected_abort
        multi_segment.mark_aborted_no_grad(
            sample, session_id=session_id, instance_id=instance_id
        )
        _set_status(sample, "ABORTED")
        return sample
    finally:
        if initialized and paddock is not None:
            if strict_errors or _await_paddock_termination():
                # Release pooled slots before returning to Slime's synchronous
                # reward/token conversion.  Besides avoiding event-loop
                # starvation, this places cleanup on the trajectory's natural
                # backpressure path: a full pool must release before the next
                # wave can make progress anyway.
                await terminate_paddock_best_effort(
                    paddock,
                    session_id=session_id,
                    env_args=env_args,
                )
            else:
                schedule_terminate_paddock(
                    paddock,
                    session_id=session_id,
                    env_args=env_args,
                )
