"""Synchronous rollout entrypoint for Dressage on colocate setups.

Used when actor and sglang share the same GPUs (e.g. qwen3.5-35B-A3B on
8xH100 with `--colocate`). Mirrors the dressage retry / empty-batch /
failure-summary semantics of `fully_async_rollout`, but runs to completion
per `rollout_id` so the framework can offload the sglang engine before
training kicks in.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dressage.rollout.case_failure import is_case_failure_no_grad
from dressage.config import proxy_url
from dressage.proxy.proxy_client import ProxyClient
from dressage.rollout.fully_async_rollout import (
    _allow_empty_train_batch,
    _flatten_multi_segment_result,
    _group_failure_summary,
    _group_has_trainable_tokens,
    _increment_retry,
    _is_aborted_group,
    _mark_no_grad_failed,
    _retry_count,
)
from dressage.rollout.generate.runtime import generate_group

logger = logging.getLogger(__name__)

try:
    import sglang_router
    from packaging.version import parse

    from slime.backends.sglang_utils.server_control import wait_servers_until_stably_idle
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
    from slime.utils.async_utils import run
    from slime.utils.http_utils import get
except ImportError:
    GenerateState = None  # type: ignore[assignment]
    generate_and_rm_group = None  # type: ignore[assignment]
    sglang_router = None  # type: ignore[assignment]
    wait_servers_until_stably_idle = None  # type: ignore[assignment]
    RolloutFnTrainOutput = None  # type: ignore[assignment]

    def run(coro):  # type: ignore[no-redef]
        return asyncio.run(coro)

from dressage.rollout.multi_segment import (
    compute_multi_segment_metrics,
    compute_prewarm_metrics,
)


def _max_retries() -> int:
    return int(os.environ.get("DRESSAGE_ROLLOUT_MAX_RETRIES", "2"))


def _max_replacement_groups(target: int) -> int:
    return int(
        os.environ.get(
            "DRESSAGE_SYNC_MAX_REPLACEMENT_GROUPS",
            str(max(4, target)),
        )
    )


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _strict_group_failure(group: list[Any], expected_trajectories: int) -> str | None:
    """Return a reason when a generated GRPO group is not a complete G group."""

    if _is_aborted_group(group):
        return "contains an aborted trajectory"
    by_rollout: dict[Any, list[Any]] = {}
    for sample in group:
        rollout_id = getattr(sample, "rollout_id", None)
        if rollout_id is None:
            return "contains a sample without rollout_id"
        by_rollout.setdefault(rollout_id, []).append(sample)
    if len(by_rollout) != expected_trajectories:
        return (
            f"contains {len(by_rollout)} trajectories; "
            f"expected {expected_trajectories}"
        )
    empty = [
        rollout_id
        for rollout_id, samples in by_rollout.items()
        if not _group_has_trainable_tokens(samples) and not all(is_case_failure_no_grad(s) for s in samples)
    ]
    if empty:
        return f"contains {len(empty)} trajectories without trainable tokens"
    return None


async def _wait_for_rollout_servers_to_drain(args: Any) -> None:
    if (
        wait_servers_until_stably_idle is None
        or sglang_router is None
        or not hasattr(args, "sglang_router_ip")
        or not hasattr(args, "sglang_router_port")
    ):
        return
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"{router}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"{router}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    await wait_servers_until_stably_idle(urls)


async def _resume_proxy_after_sync_boundary() -> None:
    client = ProxyClient(proxy_url())
    try:
        response = await client.resume_rollout(reason="sync_rollout_start")
    finally:
        await client.close()
    if response.get("status") in {"backend_not_ready", "shutting_down"}:
        raise RuntimeError(f"Dressage proxy could not resume: {response}")


async def _pause_proxy_for_natural_drain() -> None:
    timeout_seconds = float(
        os.environ.get("DRESSAGE_PROXY_DRAIN_TIMEOUT_SEC", "600")
    )
    client = ProxyClient(proxy_url(), timeout=timeout_seconds + 30.0)
    try:
        response = await client.pause_rollout(
            reason="sync_rollout_batch_boundary",
            timeout_seconds=timeout_seconds,
            mode="drain",
        )
    finally:
        await client.close()
    if not response.get("quiesced", False):
        raise RuntimeError(
            "Dressage proxy failed to drain naturally before colocated "
            f"training: {response}"
        )
    logger.info("Dressage proxy is naturally drained: %s", response)


async def _submit_group(
    args: Any,
    group: list[Any],
    state: Any,
    pendings: set[asyncio.Task],
    task_to_group: dict[asyncio.Task, list[Any]],
) -> None:
    if generate_and_rm_group is None:
        raise RuntimeError("slime.rollout.sglang_rollout.generate_and_rm_group is unavailable")
    operation = generate_group
    task = asyncio.create_task(
        operation(
            generate_and_rm_group,
            args,
            group,
            sampling_params=state.sampling_params.copy(),
        )
    )
    pendings.add(task)
    task_to_group[task] = group


async def _run_sync_rollout(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
) -> list[list[Any]]:
    if GenerateState is None or generate_and_rm_group is None:
        raise RuntimeError(
            "Dressage sync rollout requires slime.rollout.sglang_rollout to be importable"
        )

    target = int(getattr(args, "rollout_batch_size", 1))
    max_retries = _max_retries()
    max_replacements = _max_replacement_groups(target)
    strict_complete_groups = _env_flag("DRESSAGE_REQUIRE_COMPLETE_GROUPS")
    allow_partial_final_batch = _env_flag("DRESSAGE_ALLOW_PARTIAL_FINAL_BATCH")
    if strict_complete_groups and max_replacements != 0:
        raise ValueError(
            "DRESSAGE_REQUIRE_COMPLETE_GROUPS=1 requires "
            "DRESSAGE_SYNC_MAX_REPLACEMENT_GROUPS=0"
        )
    replacements_used = 0
    state = GenerateState(args)
    data: list[list[Any]] = []
    pendings: set[asyncio.Task] = set()
    task_to_group: dict[asyncio.Task, list[Any]] = {}

    # The previous synchronous batch keeps proxy admission closed while the
    # colocated model is trained/offloaded.  Reopen only after Slime has made
    # the SGLang workers ready for this rollout.
    await _resume_proxy_after_sync_boundary()

    groups = data_buffer.get_samples(target)
    if not groups:
        raise RuntimeError(
            f"data_buffer.get_samples({target}) returned no groups"
        )
    if len(groups) < target and not allow_partial_final_batch:
        raise RuntimeError(
            f"data_buffer.get_samples({target}) returned {len(groups)} groups; "
            "Dressage sync rollout submits the whole batch one-shot and does not oversample."
        )
    effective_target = len(groups)
    for group in groups:
        await _submit_group(args, group, state, pendings, task_to_group)

    while pendings:
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group_for_task = task_to_group.pop(task)
            error: BaseException | None = None
            result_group: list[Any] | None = None
            try:
                result_group = task.result()
            except BaseException as exc:  # noqa: BLE001 - mirror fully_async behavior
                error = exc

            if error is not None and any((s.metadata or {}).get('h0_first_rescue') or
                    (s.metadata or {}).get('h0_retry_on_zero') for s in group_for_task):
                # The rescue runner has already handled ordinary case failures.
                # Identity/checkpoint/storage errors must never become zero rewards.
                for pending in pendings:
                    pending.cancel()
                await asyncio.gather(*pendings, return_exceptions=True)
                raise error

            if result_group is not None:
                result_group = _flatten_multi_segment_result(result_group)

            no_trainable_tokens = (
                result_group is not None
                and not _group_has_trainable_tokens(result_group)
                and not all(is_case_failure_no_grad(s) for s in result_group)
            )
            if no_trainable_tokens:
                for sample in result_group:
                    metadata = getattr(sample, "metadata", None)
                    if isinstance(metadata, dict):
                        metadata["rollout_error"] = (
                            "rollout group contained no trainable response tokens"
                        )
            strict_failure = (
                _strict_group_failure(result_group, len(group_for_task))
                if strict_complete_groups and result_group is not None
                else None
            )
            failed = (
                error is not None
                or _is_aborted_group(result_group or group_for_task)
                or no_trainable_tokens
                or strict_failure is not None
            )
            if not failed:
                data.append(result_group)
                continue

            summary = _group_failure_summary(
                result_group if result_group is not None else group_for_task, error
            )
            if strict_failure is not None:
                summary = f"{summary}; strict_group_failure={strict_failure}"
            if _retry_count(group_for_task) < max_retries and getattr(error, "retryable", True):
                _increment_retry(group_for_task)
                logger.warning(
                    "resubmitting rollout group for retry (attempt %d/%d): %s",
                    _retry_count(group_for_task),
                    max_retries,
                    summary,
                )
                await _submit_group(args, group_for_task, state, pendings, task_to_group)
            else:
                replacement: list[Any] | None = None
                if replacements_used < max_replacements:
                    replacement_groups = data_buffer.get_samples(1)
                    if replacement_groups:
                        replacement = replacement_groups[0]
                if replacement is not None:
                    replacements_used += 1
                    logger.error(
                        "rollout group exhausted retries and will be replaced "
                        "(%d/%d): %s",
                        replacements_used,
                        max_replacements,
                        summary,
                    )
                    await _submit_group(
                        args,
                        replacement,
                        state,
                        pendings,
                        task_to_group,
                    )
                else:
                    logger.error(
                        "rollout group exhausted retries and no replacement is "
                        "available; it will be marked failed: %s",
                        summary,
                    )
                    # Retry inputs intentionally remain pristine. Preserve the
                    # failed attempt's diagnostics instead of replacing them
                    # with an untouched input lacking its runtime identity.
                    data.append(_mark_no_grad_failed(
                        result_group if result_group is not None else group_for_task, error))

    # A proxy/client disconnect can finish a Dressage task while its original
    # SGLang /generate request is still alive.  Atomically close new proxy
    # admission, let already-admitted requests finish without aborting Mamba,
    # then require stable worker-idle observations before model offload.
    await _pause_proxy_for_natural_drain()
    await _wait_for_rollout_servers_to_drain(args)

    state.reset()

    data = sorted(data, key=lambda group: getattr(group[0], "index", 0))
    if strict_complete_groups:
        strict_failures = [
            reason
            for group in data
            if (reason := _strict_group_failure(group, int(getattr(args, "n_samples_per_prompt", 1))))
            is not None
        ]
        if len(data) != effective_target or strict_failures:
            diagnostics = [_group_failure_summary(group) for group in data
                           if _strict_group_failure(group, int(getattr(args, "n_samples_per_prompt", 1)))][:3]
            raise RuntimeError(
                "Dressage strict rollout batch is incomplete; refusing optimizer step: "
                f"groups={len(data)}/{effective_target} failures={strict_failures[:3]}; "
                f"diagnostics={diagnostics}"
            )
    if (not _allow_empty_train_batch()
            and not all(is_case_failure_no_grad(s) for group in data for s in group)) and not any(
        _group_has_trainable_tokens(group) for group in data
    ):
        summaries = [_group_failure_summary(group) for group in data[: min(3, len(data))]]
        raise RuntimeError(
            "Dressage sync rollout produced no trainable samples; "
            "refusing to train on failed placeholder samples. "
            f"First failures: {' | '.join(summaries)}. "
            "Set DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1 to keep the previous behavior."
        )

    return data


def generate_rollout_sync(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
):
    if evaluation:
        raise ValueError("Dressage sync rollout does not support evaluation mode")
    data = run(_run_sync_rollout(args, rollout_id, data_buffer))
    flat = [sample for group in data for sample in group]
    metrics: dict[str, Any] = compute_multi_segment_metrics(flat)
    metrics.update(compute_prewarm_metrics(flat))
    metrics["rollout/task_groups"] = float(len(data))
    metrics["rollout/trajectories"] = float(
        len({getattr(sample, "rollout_id", None) for sample in flat})
    )
    if RolloutFnTrainOutput is None:
        return data
    return RolloutFnTrainOutput(samples=data, metrics=metrics)
