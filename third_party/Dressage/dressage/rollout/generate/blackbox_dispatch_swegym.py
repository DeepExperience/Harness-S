"""SWE-Gym-specific blackbox rollout generation hook."""

from __future__ import annotations

import logging
import asyncio
import uuid
from typing import Any

from dressage.recipes.swegym.deadlines import bounded, seconds, cancel_session
from dressage.rollout.case_failure import mark_case_failure_no_grad
from dressage.recipes.swegym.diagnostics import policy_evaluation_failure
from dressage.config import proxy_url
from dressage.recipes.swegym.evaluator import (
    execute_fresh_swegym_eval,
    is_environment_request_error,
    record_agent_environment_exception,
    use_fresh_swegym_eval,
)
from dressage.recipes.swegym.command_contract import validate_command_schedule
from dressage.recipes.swegym.integrity import (
    reset_integrity_metadata,
    scan_trajectory_integrity,
    use_swegym_integrity,
)
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
)
from dressage.rollout import multi_segment
from dressage.rollout.artifacts.samples import (
    instance_id as _instance_id,
    set_status as _set_status,
)
from dressage.rollout.artifacts.writer import DEFAULT_WRITER as _ARTIFACT_WRITER
from dressage.rollout.generate.runtime import (
    get_paddock_from_env,
    get_proxy_client,
    maybe_await,
    paddock_env_args_from_metadata,
)

logger = logging.getLogger(__name__)


def _chat_messages_from_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    return [{"role": "user", "content": str(prompt)}]


def _ensure_blackbox_session_id(sample: Any) -> str:
    session_id = getattr(sample, "session_id", None)
    if session_id is None:
        session_id = str(uuid.uuid4())
    session_id = str(session_id)

    if not session_id.startswith("bbs-"):
        session_id = f"bbs-{session_id}"
        sample.session_id = session_id

    return session_id


def _backend_options_for_register(
    *,
    args: Any,
    metadata: dict[str, Any],
    blackbox_type: str,
) -> Any:
    backend_options = metadata.get("backend_options")
    return merge_backend_options(blackbox_type, backend_options, args=args)


async def _generate_once(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Run one blackbox sandbox rollout and write proxy data back to Sample."""
    del sampling_params
    if evaluation:
        raise ValueError("blackbox_dispatch_swegym does not support evaluation mode")

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata.pop("blackbox_error", None)
    metadata.pop("blackbox_error_log_path", None)
    metadata.pop("blackbox_expected_abort", None)
    metadata.pop("blackbox_agent_early_stop", None)
    metadata.pop("blackbox_agent_early_stop_kind", None)
    swegym_integrity_enabled = use_swegym_integrity(metadata)
    if swegym_integrity_enabled:
        reset_integrity_metadata(metadata)
    metadata["execute_cmds"] = []
    session_id = _ensure_blackbox_session_id(sample)
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
    state = None
    initialized = False
    agent_response = ""
    try:
        execute_cmd_schedule = validate_command_schedule(metadata)
        backend_options = _backend_options_for_register(
            args=args,
            metadata=metadata,
            blackbox_type=blackbox_type,
        )
        paddock = get_paddock_from_env(allow_whitebox_mode=False)
        proxy_client = get_proxy_client()
        state = await maybe_await(
            paddock.init(
                session_id,
                metadata.get("env_type"),
                env_args,
            )
        )
        initialized = True
        if not hasattr(paddock, "register_agent"):
            raise TypeError(
                f"{type(paddock).__name__} does not implement register_agent"
            )
        await maybe_await(
            paddock.register_agent(
                state,
                instance_id=instance_id,
                session_id=session_id,
                router_url=proxy_url(),
                blackbox_type=blackbox_type,
                backend_options=backend_options,
            )
        )
        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="before_agent",
        )
        call_payload: Any = None
        call_succeeded = False
        agent_environment_failed = False
        try:
            call_payload = await bounded(maybe_await(
                paddock.call_agent(
                    state,
                    session_id=session_id,
                    messages=_chat_messages_from_prompt(sample.prompt),
                    metadata={"source": "dressage", **metadata},
                )
            ), "agent")
            call_succeeded = True
        except Exception as exc:
            from dressage.rollout.case_failure import is_global_failure
            if is_global_failure(exc):
                raise
            if expected_abort_from_call_agent_exception(exc) is not None:
                raise
            agent_failure = failure_from_call_agent_exception(exc)
            if agent_failure is not None:
                record_agent_failure_metadata(metadata, agent_failure)
            if agent_failure is not None and agent_failure.kind == "backend_process_exit_with_partial_trajectory":
                # Process exits are case failures even if a partial patch exists.
                # The proxy may not have seen a first request; do not assume tokens.
                metadata["swegym_runtime_failure"] = policy_evaluation_failure(error=exc)
                agent_environment_failed = True
            elif agent_failure is not None and agent_failure.kind in HARVESTABLE_AGENT_ERRORS:
                record_agent_early_stop_metadata(metadata, agent_failure)
                logger.info(
                    "harvesting blackbox rollout after agent early stop: "
                    "session_id=%s kind=%s", session_id, agent_failure.kind,
                )
            elif expected_abort_from_call_agent_exception(exc) is not None:
                raise  # Shared preemption/resume semantics must remain intact.
            elif is_environment_request_error(exc):
                record_agent_environment_exception(metadata, exc)
                metadata["swegym_runtime_failure"] = policy_evaluation_failure(error=exc)
                agent_environment_failed = True
                logger.warning("harvesting SWE-Gym environment failure as zero: session_id=%s error=%s",
                               session_id, exc)
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

        if agent_environment_failed:
            pass  # Finalize/drain below; empty trajectories still fail validation.
        elif use_fresh_swegym_eval(metadata):
            await execute_fresh_swegym_eval(
                paddock,
                state,
                metadata,
                schedule=execute_cmd_schedule,
                source_session_id=session_id,
                instance_id=instance_id,
                env_type=metadata.get("env_type"),
                env_args=env_args,
                blackbox_type=blackbox_type,
                backend_options=backend_options,
            )
        else:
            await execute_blackbox_cmds_for_stage(
                paddock,
                state,
                metadata,
                schedule=execute_cmd_schedule,
                session_id=session_id,
                stage="after_agent",
            )
        try:
            await proxy_client.finalize_session(
                session_id, instance_id=instance_id, label=getattr(sample, "label", None)
            )
        except Exception as exc:
            if not is_environment_request_error(exc):
                raise
            metadata.setdefault("swegym_runtime_failure", policy_evaluation_failure(error=exc))
            metadata["trajectory_finalize_error"] = _exception_summary(exc)
        # Even a failed finalize can have already committed segments. Recover
        # those exact tokens before considering an explicitly masked zero.
        try:
            trajectory_payload = await proxy_client.read_trajectory(
                trajectory_id=session_id, instance_id=instance_id, drain=True,
            )
        except Exception as exc:
            if not is_environment_request_error(exc):
                raise
            metadata.setdefault("swegym_runtime_failure", policy_evaluation_failure(error=exc))
            metadata["trajectory_read_error"] = _exception_summary(exc)
            trajectory_payload = {"data": []}
        if not trajectory_payload.get("data") and metadata.get("swegym_runtime_failure"):
            metadata["swegym_reward"] = {**metadata["swegym_runtime_failure"], "reward": 0.0, "raw_reward": 0.0}
            logger.warning("SWE-Gym case failed without proxy tokens; retaining audited zero without gradient: %s", session_id)
            return mark_case_failure_no_grad(sample, session_id=session_id,
                                             diagnostics=metadata["swegym_reward"])
        if swegym_integrity_enabled:
            scan_trajectory_integrity(trajectory_payload, metadata)
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
        return result
    except Exception as exc:
        from dressage.rollout.case_failure import is_global_failure
        if is_global_failure(exc):
            raise
        expected_abort = expected_abort_from_call_agent_exception(exc)
        if expected_abort is None:
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
            from dressage.rollout.case_failure import zero_case_failures
            if zero_case_failures() and initialized:
                # Preserve any real generated tokens even when patch/grader setup failed.
                metadata["swegym_runtime_failure"] = policy_evaluation_failure(error=exc)
                try:
                    await proxy_client.finalize_session(session_id, instance_id=instance_id,
                                                        label=getattr(sample, "label", None))
                except Exception as finalize_error:
                    if is_global_failure(finalize_error):
                        raise
                    metadata["trajectory_finalize_error"] = _exception_summary(finalize_error)
                try:
                    payload = await proxy_client.read_trajectory(trajectory_id=session_id,
                                                                 instance_id=instance_id, drain=True)
                except Exception as read_error:
                    if is_global_failure(read_error):
                        raise
                    metadata["trajectory_read_error"] = _exception_summary(read_error)
                    payload = {"data": []}
                if payload.get("data"):
                    if swegym_integrity_enabled:
                        scan_trajectory_integrity(payload, metadata)
                    return multi_segment.expand_segments_to_samples(sample, payload["data"], args=args,
                        agent_response=agent_response, session_id=session_id, instance_id=instance_id)
            record_blackbox_abort_for_retry(metadata, session_id, exc)
        else:
            metadata["blackbox_expected_abort"] = expected_abort
        multi_segment.mark_aborted_no_grad(
            sample, session_id=session_id, instance_id=instance_id
        )
        _set_status(sample, "ABORTED")
        return sample
    finally:
        if seconds('case') and initialized:
            # Stop the exact outstanding proxy request before releasing the
            # corresponding sandbox. Engine-side idle checks remain required.
            try:
                await cancel_session(proxy_client, session_id)
            except Exception as cleanup_error:
                metadata['request_cleanup_error'] = str(cleanup_error)
        if initialized and paddock is not None:
            schedule_terminate_paddock(
                paddock,
                session_id=session_id,
                env_args=env_args,
            )


async def generate(args, sample, sampling_params, evaluation=False):
    """Bounded retries at the trajectory boundary preserve all task/clone IDs."""
    import asyncio
    from copy import deepcopy
    from dressage.rollout.case_failure import zero_case_failures, is_global_failure
    if not zero_case_failures():
        return await _generate_once(args, sample, sampling_params, evaluation)
    # Validate the recipe before entering the case failure boundary.
    validate_command_schedule(sample.metadata)
    attempts = []
    deadline = asyncio.get_running_loop().time() + seconds('case') if seconds('case') else None
    for attempt in range(3):
        candidate = deepcopy(sample)
        candidate.session_id = 'bbs-' + uuid.uuid4().hex
        try:
            remaining = deadline - asyncio.get_running_loop().time() if deadline else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError('SWE-Gym case wall-clock timeout across retries')
            result = await bounded(_generate_once(args, candidate, sampling_params, evaluation), 'case', timeout=remaining)
        except TimeoutError as exc:
            diag = policy_evaluation_failure(error=exc)
            candidate.metadata.update(swegym_reward=diag, case_retry_history=attempts, case_timed_out=True)
            # The cancelled coroutine may already have finalized real segments.
            try:
                client = get_proxy_client()
                await cancel_session(client, candidate.session_id)
                payload = await asyncio.wait_for(client.read_trajectory(
                    trajectory_id=candidate.session_id, instance_id=_instance_id(candidate), drain=True), 10)
                if payload.get('data'):
                    candidate.metadata['swegym_runtime_failure'] = diag
                    if use_swegym_integrity(candidate.metadata):
                        scan_trajectory_integrity(payload, candidate.metadata)
                    return multi_segment.expand_segments_to_samples(candidate, payload['data'], args=args,
                        agent_response='', session_id=candidate.session_id, instance_id=_instance_id(candidate))
            except Exception as recovery_error:
                if is_global_failure(recovery_error):
                    raise
                candidate.metadata['timeout_recovery_error'] = str(recovery_error)
            return mark_case_failure_no_grad(candidate, session_id=candidate.session_id, diagnostics=diag)
        segments = result if isinstance(result, list) else [result]
        aborted = any(getattr(s.status, 'name', '') == 'ABORTED' for s in segments)
        if not aborted:
            for item in segments:
                item.metadata['case_retry_history'] = attempts
            return result
        error = candidate.metadata.get('blackbox_error') or candidate.metadata.get('blackbox_agent_error_message') or 'trajectory execution failed'
        if is_global_failure(RuntimeError(str(error))):
            raise RuntimeError(str(error))
        attempts.append({'attempt': attempt + 1, 'session_id': candidate.session_id, 'error': error})
        if seconds('case') and any(word in str(error).lower() for word in ('timeout', 'timed out', 'http 504')):
            diag = policy_evaluation_failure(error=TimeoutError(str(error)))
            candidate.metadata.update(swegym_reward=diag, case_retry_history=attempts, case_timed_out=True)
            return mark_case_failure_no_grad(candidate, session_id=candidate.session_id, diagnostics=diag)
        if attempt < 2:
            await asyncio.sleep(attempt + 1)
            continue
        diag = policy_evaluation_failure(error=RuntimeError(str(error)))
        candidate.metadata.update(swegym_reward=diag, case_retry_history=attempts)
        return mark_case_failure_no_grad(candidate, session_id=candidate.session_id, diagnostics=diag)
