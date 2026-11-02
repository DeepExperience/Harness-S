"""Docker provider for project-local SWE-Gym blackbox rollouts."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import shlex
import uuid
from pathlib import Path
from typing import Any

import httpx
from dressage.recipes.swegym.runtime import docker_runtime_args, tool_path, task_python_path, is_task_image, TOOL_ENV_PROBE

from dressage.sandbox.types import (
    CommandResult,
    SandboxEndpoint,
    SandboxLease,
    SandboxSpec,
)


# Linux limits the size of each execve(2) argument (MAX_ARG_STRLEN), which is
# commonly 128 KiB even when ARG_MAX is much larger.  Fresh SWE-Gym evaluation
# commands can contain a compressed model patch and legitimately exceed that
# limit.  Keep ordinary commands inspectable in `docker exec ... bash -lc`, but
# stream large shell programs over stdin so their size is not part of argv.
_MAX_INLINE_SHELL_BYTES = 64 * 1024
logger = logging.getLogger(__name__)


def checkpoint_harness_mount_args(project_root: Path) -> list[str]:
    """Expose immutable runtime paths embedded in a trained data fingerprint."""
    run_value = os.environ.get("HAPT_JOINT_RUN_ROOT")
    if not run_value:
        return []
    run = Path(run_value).resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text())
    checkpoint = manifest.get("checkpoint_eval")
    if manifest.get("execution_mode") != "checkpoint_eval" or not checkpoint:
        return []
    source = Path(checkpoint["source_run_root"]).resolve()
    allowed = (project_root / "outputs/policy-training").resolve()
    if not source.is_relative_to(allowed):
        raise ValueError("checkpoint evaluation source is outside policy-training outputs")
    versions = source / "harness-versions"
    if not versions.is_dir():
        raise ValueError("checkpoint evaluation historical Harness directory is missing")
    return ["--mount", f"type=bind,src={versions},dst={versions},readonly"]


@dataclass(slots=True)
class _DockerPoolSlot:
    key: tuple[Any, ...]
    container_name: str
    image: str
    ip_address: str
    service_port: int
    http_client: httpx.AsyncClient
    uses: int = 0
    leased_to: str | None = None


class DockerSandboxProvider:
    """Create one exact Docker container per trajectory.

    Agent containers run BlackboxServer on the project-internal network.
    Fresh evaluator containers have ``--network none`` and no BlackboxServer;
    ``DockerBlackboxAgentPaddock`` executes their commands directly.
    """

    name = "docker"

    def __init__(
        self,
        *,
        docker_bin: str | None = None,
        project_root: str | Path | None = None,
        network: str | None = None,
    ) -> None:
        inferred_root = Path(__file__).resolve().parents[6]
        self.project_root = Path(
            project_root or os.environ.get("HAPT_PROJECT_ROOT") or inferred_root
        ).resolve()
        self.docker_bin = (
            docker_bin
            or os.environ.get("HAPT_DOCKER_BIN")
            or self._default_docker_bin()
        )
        self.network = network or os.environ.get(
            "HAPT_SANDBOX_NETWORK", "hapt-internal"
        )
        self.docker_host = os.environ.get(
            "DOCKER_HOST", "unix:///var/run/docker.sock"
        )
        self._leases: dict[str, SandboxLease] = {}
        self.pool_size = _env_int("DRESSAGE_DOCKER_POOL_SIZE", 0, min_value=0)
        self.health_timeout_sec = _env_float(
            "DRESSAGE_DOCKER_HEALTH_TIMEOUT_SEC", 30.0, min_value=1.0
        )
        self.pool_max_uses = _env_int(
            "DRESSAGE_DOCKER_POOL_MAX_USES", 50, min_value=1
        )
        self.pool_reset_timeout_sec = _env_float(
            "DRESSAGE_DOCKER_POOL_RESET_TIMEOUT_SEC", 15.0, min_value=0.1
        )
        self.pool_reset_rpc_timeout_sec = _env_float(
            "DRESSAGE_DOCKER_POOL_RESET_RPC_TIMEOUT_SEC",
            min(10.0, self.pool_reset_timeout_sec),
            min_value=0.1,
        )
        self.pool_recovery_timeout_sec = _env_float(
            "DRESSAGE_DOCKER_POOL_RECOVERY_TIMEOUT_SEC", 30.0, min_value=0.1
        )
        # A rollout wave can finish many trajectories together.  Bounding the
        # number of simultaneous process-group aborts and fresh Docker-bridge
        # connections prevents that completion burst from turning into a
        # reset-timeout/recycle storm.  The gate is provider-local, so paired
        # Harness arms remain isolated and can both make progress.
        self.pool_reset_concurrency = _env_int(
            "DRESSAGE_DOCKER_POOL_RESET_CONCURRENCY",
            min(8, self.pool_size) if self.pool_size > 0 else 8,
            min_value=1,
        )
        pool_run_label = (
            os.environ.get("DRESSAGE_DOCKER_POOL_RUN_ID")
            or os.environ.get("DRESSAGE_RUN_NAME")
            or f"pid-{os.getpid()}"
        )
        # Docker container names benefit from a compact identifier, but the
        # cleanup label must retain the exact run identifier exported by the
        # launcher.  In particular, DSH and Claude Code run names often share
        # more than 24 prefix characters.  Reusing a naively truncated value
        # for the label lets one arm's EXIT trap delete the other arm's pool.
        self.pool_run_label = str(pool_run_label)
        self.pool_run_id = _safe_name(self.pool_run_label, max_length=24)
        http_connection_limit = max(32, self.pool_size * 2)
        # Health checks establish one connection per pool slot.  Retain those
        # connections for abort/status reset RPCs instead of opening a burst of
        # fresh TCP connections across the Docker bridge at the end of every
        # rollout wave.  The old per-reset client caused connect timeouts once
        # 24 trajectories completed together.
        self._http_client = httpx.AsyncClient(
            trust_env=False,
            limits=httpx.Limits(
                max_connections=http_connection_limit,
                max_keepalive_connections=http_connection_limit,
                # BlackboxServer uses Uvicorn's five-second default.  Expire
                # client connections first instead of racing a stale FIN.
                keepalive_expiry=4.0,
            ),
        )
        self._pool_slots: list[_DockerPoolSlot] = []
        self._pool_creating = 0
        self._pool_condition = asyncio.Condition()
        self._pool_reset_semaphore = asyncio.Semaphore(
            self.pool_reset_concurrency
        )
        self._pool_leases: dict[str, _DockerPoolSlot] = {}
        self._retired_http_close_tasks: set[asyncio.Task[Any]] = set()

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        image = str(
            spec.env_args.get("sandbox_image")
            or os.environ.get("DRESSAGE_SANDBOX_DEFAULT_IMAGE")
            or ""
        ).strip()
        if not image:
            raise ValueError("Docker sandbox requires env_args.sandbox_image")
        await self._ensure_image(image)
        evaluator = spec.trajectory_id.startswith("swegym-eval-") or (
            spec.metadata.get("role") == "evaluator"
        )
        if self.pool_size > 0 and not evaluator:
            return await self._create_pooled(spec, image)
        lease = await self._create_exact(spec, image=image, evaluator=evaluator)
        self._leases[spec.trajectory_id] = lease
        return lease

    async def _create_exact(
        self,
        spec: SandboxSpec,
        *,
        image: str,
        evaluator: bool,
        container_name: str | None = None,
        pooled: bool = False,
    ) -> SandboxLease:
        container_name = container_name or self._container_name(
            spec.trajectory_id, evaluator=evaluator
        )
        command = [
            self.docker_bin,
            "run",
            "--detach",
            "--name",
            container_name,
            "--network",
            "none" if evaluator else self.network,
            "--workdir",
            str(spec.env_args.get("workdir") or "/testbed"),
            "--label",
            f"hapt.dressage.pool_run={self.pool_run_label}",
        ]
        # Large hosts expose hundreds of logical CPUs.  BLAS libraries inside
        # task images otherwise create more worker threads than their builds
        # support; at rollout concurrency this can oversubscribe the host and
        # has caused an official Terminal solution to segfault.  Docker exec
        # inherits these container-level values for agent and verifier tools.
        for key in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            command.extend(["--env", f"{key}=1"])
        # Ownership also covers per-task agent/evaluator containers. The
        # shared supervisor can then clean them after an interrupted rollout.
        command.extend(docker_runtime_args(image, self.project_root, self._node_root() / "bin", evaluator=evaluator))
        if evaluator:
            command.extend([image, "sleep", "infinity"])
        else:
            service_port = self._service_port(spec)
            command.extend(
                self._agent_mount_args(
                    spec.env_args.get("sandbox_readonly_mounts")
                )
            )
            command.extend(
                [
                    "--env",
                    f"BBS_PORT={service_port}",
                    "--env",
                    "BBS_HOST=0.0.0.0",
                    "--env",
                    "BBS_RUNTIME_ROOT=/tmp/hapt-blackbox-server",
                    "--env",
                    "BBS_BACKEND_TIMEOUT=1800",
                    "--env",
                    "HAPT_DSH_NODE_EXECUTABLE=" + str(self._node_root() / "bin/node"),
                    "--env",
                    "BBS_EXECUTE_CMD_TIMEOUT=1800",
                    "--env",
                    "PYTHONUNBUFFERED=1",
                    "--env",
                    "PYTHONPATH=" + task_python_path(image, "/opt/hapt-runtime/third_party/Dressage:/opt/hapt-runtime/third_party/Dressage/blackbox_server"),
                    "--env",
                    "PATH=" + tool_path(image, self._node_root() / "bin"),
                    image,
                    "/opt/harness-s-python",
                    # Do not prepend /testbed to the BlackboxServer import
                    # path.  Repositories such as pydantic contain a top-level
                    # package whose name can otherwise shadow FastAPI's
                    # runtime dependency before the server starts.  Python's
                    # process-local -P flag fixes only the control server; it
                    # is not inherited by agent/tool subprocesses, which must
                    # continue to execute with normal /testbed semantics.
                    "-P",
                    "-m",
                    "blackbox_server.main",
                ]
            )
        result = await self._run(command, timeout=1800.0)
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker sandbox create failed for {image}: {result.stderr[-4000:]}"
            )

        metadata: dict[str, Any] = {
            "container_name": container_name,
            "image": image,
            "network": "none" if evaluator else self.network,
            "evaluator_only": evaluator,
            "pooled": pooled,
        }
        if is_task_image(image):
            try:
                probe = await self._run(
                    [self.docker_bin, "exec", container_name, "bash", "-c",
                     "python -c " + shlex.quote(TOOL_ENV_PROBE)], timeout=60.0)
                if probe.returncode:
                    raise RuntimeError(f"SWE-Gym task Python probe failed: {probe.stderr[-2000:]}")
                metadata["tool_environment"] = json.loads(probe.stdout)
            except BaseException:
                await self._run([self.docker_bin, "rm", "--force", container_name], timeout=300.0)
                raise
        endpoints: dict[str, SandboxEndpoint] = {}
        capabilities = {"command", "file"}
        if not evaluator:
            ip_address = await self._container_ip(container_name)
            service_port = self._service_port(spec)
            endpoint = SandboxEndpoint(url=f"http://{ip_address}:{service_port}")
            try:
                await self._wait_for_health(endpoint.url, timeout=self.health_timeout_sec)
            except Exception as error:
                diagnostics = await self._container_startup_diagnostics(
                    container_name
                )
                await self._run(
                    [self.docker_bin, "rm", "--force", container_name],
                    timeout=300.0,
                )
                raise RuntimeError(
                    f"{error}; container={container_name}; {diagnostics}"
                ) from error
            endpoints["blackbox"] = endpoint
            for service in spec.services:
                endpoints[service.name] = SandboxEndpoint(
                    url=f"http://{ip_address}:{service.port}"
                )
            capabilities.add("public_url")
            metadata["ip_address"] = ip_address

        lease = SandboxLease(
            trajectory_id=spec.trajectory_id,
            provider=self.name,
            sandbox_id=container_name,
            endpoints=endpoints,
            capabilities=capabilities,
            metadata=metadata,
            raw={"container_name": container_name},
        )
        return lease

    async def _create_pooled(self, spec: SandboxSpec, image: str) -> SandboxLease:
        key = self._pool_key(spec, image)
        create_slot = False
        slot: _DockerPoolSlot | None = None
        while slot is None:
            async with self._pool_condition:
                slot = next(
                    (
                        candidate
                        for candidate in self._pool_slots
                        if candidate.key == key and candidate.leased_to is None
                    ),
                    None,
                )
                if slot is not None:
                    slot.leased_to = spec.trajectory_id
                    break
                if len(self._pool_slots) + self._pool_creating < self.pool_size:
                    self._pool_creating += 1
                    create_slot = True
                    break
                await self._pool_condition.wait()

        if create_slot:
            try:
                container_name = self._pool_container_name()
                created = await self._create_exact(
                    spec,
                    image=image,
                    evaluator=False,
                    container_name=container_name,
                    pooled=True,
                )
                slot = _DockerPoolSlot(
                    key=key,
                    container_name=container_name,
                    image=image,
                    ip_address=str(created.metadata["ip_address"]),
                    service_port=self._service_port(spec),
                    http_client=self._new_pool_http_client(),
                    leased_to=spec.trajectory_id,
                )
                async with self._pool_condition:
                    self._pool_slots.append(slot)
            finally:
                async with self._pool_condition:
                    self._pool_creating -= 1
                    self._pool_condition.notify_all()

        assert slot is not None
        lease = self._pooled_lease(spec, slot)
        self._leases[spec.trajectory_id] = lease
        self._pool_leases[spec.trajectory_id] = slot
        return lease

    async def terminate(self, lease: SandboxLease | str) -> dict[str, Any]:
        trajectory_id = lease if isinstance(lease, str) else lease.trajectory_id
        known = self._leases.pop(trajectory_id, None)
        slot = self._pool_leases.pop(trajectory_id, None)
        if slot is not None:
            return await self._release_pooled_slot(slot, trajectory_id)
        # A pooled lease represents a temporary claim on a shared container,
        # not ownership of the container itself.  A duplicate/late terminate
        # must never fall through to `docker rm --force` after the claim was
        # already released and potentially reassigned.
        lease_metadata = {} if isinstance(lease, str) else lease.metadata
        if bool(lease_metadata.get("pooled")) or bool(
            known is not None and known.metadata.get("pooled")
        ):
            return {
                "terminated": False,
                "trajectory_id": trajectory_id,
                "missing": True,
                "pooled": True,
            }
        container_name = (
            None if isinstance(lease, str) else lease.sandbox_id
        ) or (known.sandbox_id if known is not None else None)
        if not container_name:
            return {"terminated": False, "trajectory_id": trajectory_id, "missing": True}
        result = await self._run(
            [self.docker_bin, "rm", "--force", str(container_name)], timeout=300.0
        )
        return {
            "terminated": result.returncode == 0,
            "trajectory_id": trajectory_id,
            "container_name": container_name,
            "stderr": result.stderr,
        }

    async def _release_pooled_slot(
        self, slot: _DockerPoolSlot, trajectory_id: str
    ) -> dict[str, Any]:
        async with self._pool_reset_semaphore:
            return await self._release_pooled_slot_limited(slot, trajectory_id)

    async def _release_pooled_slot_limited(
        self, slot: _DockerPoolSlot, trajectory_id: str
    ) -> dict[str, Any]:
        async with self._pool_condition:
            current_owner = slot.leased_to
        if current_owner != trajectory_id:
            logger.warning(
                "ignored stale pooled Docker release for slot %s: "
                "trajectory_id=%s current_owner=%s",
                slot.container_name,
                trajectory_id,
                current_owner,
            )
            return {
                "terminated": False,
                "trajectory_id": trajectory_id,
                "container_name": slot.container_name,
                "pooled": True,
                "stale_release": True,
                "current_owner": current_owner,
            }
        reset_error: str | None = None
        first_error: str | None = None
        last_error: str | None = None
        attempt = 0
        reset_succeeded = False
        loop = asyncio.get_running_loop()
        recovery_deadline: float | None = None
        while True:
            attempt += 1
            if recovery_deadline is None:
                attempt_timeout = self.pool_reset_timeout_sec
            else:
                remaining = recovery_deadline - loop.time()
                if remaining <= 0:
                    break
                attempt_timeout = min(self.pool_reset_timeout_sec, remaining)
            try:
                await asyncio.wait_for(
                    self._reset_pooled_slot(slot, trajectory_id),
                    timeout=attempt_timeout,
                )
                if attempt > 1:
                    logger.info(
                        "recovered pooled Docker slot %s after %d reset retries",
                        slot.container_name,
                        attempt - 1,
                    )
                reset_succeeded = True
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # A cancelled request or transport/protocol failure can leave
                # HTTPX's keep-alive state unusable even though the container
                # is healthy.  Never carry that connection pool into the next
                # reset attempt.
                await self._renew_pool_http_client(slot)
                if first_error is None:
                    first_error = last_error
                    # Cancelling a timed-out HTTP request does not cancel the
                    # server-side abort.  Give that idempotent operation time
                    # to finish, then retry abort + exact status validation.
                    # This also absorbs short Docker-bridge ConnectTimeout
                    # bursts without declaring a clean container unusable.
                    recovery_deadline = loop.time() + self.pool_recovery_timeout_sec
                remaining = recovery_deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(0.25, remaining))

        if not reset_succeeded and last_error is not None:
            reset_error = (
                f"attempts={attempt} first={first_error}; last={last_error}"
            )

        slot.uses += 1
        recycle = reset_error is not None or slot.uses >= self.pool_max_uses
        if recycle:
            if reset_error is not None:
                logger.warning(
                    "recycling pooled Docker slot %s after reset failure: %s",
                    slot.container_name,
                    reset_error,
                )
            self._retire_http_client(slot.http_client, slot.container_name)
            removed = await self._run(
                [self.docker_bin, "rm", "--force", slot.container_name],
                timeout=300.0,
            )
            async with self._pool_condition:
                if slot in self._pool_slots:
                    self._pool_slots.remove(slot)
                slot.leased_to = None
                self._pool_condition.notify_all()
            return {
                "terminated": removed.returncode == 0,
                "trajectory_id": trajectory_id,
                "container_name": slot.container_name,
                "pooled": True,
                "recycled": True,
                "uses": slot.uses,
                "reset_error": reset_error,
                "stderr": removed.stderr,
            }

        async with self._pool_condition:
            if slot.leased_to == trajectory_id:
                slot.leased_to = None
            self._pool_condition.notify_all()
        return {
            "terminated": True,
            "trajectory_id": trajectory_id,
            "container_name": slot.container_name,
            "pooled": True,
            "recycled": False,
            "uses": slot.uses,
        }

    async def _reset_pooled_slot(
        self, slot: _DockerPoolSlot, trajectory_id: str
    ) -> None:
        base_url = f"http://{slot.ip_address}:{slot.service_port}"
        # A register response can be lost before dispatch observes success. In
        # that case the server may still expose the previous, already-finished
        # binding.  Reset the binding actually reported by this exclusively
        # leased slot instead of assuming registration reached the server.
        before = await self._pool_reset_request(
            slot,
            "GET",
            f"{base_url}/v1/status",
            phase="status_before_abort",
        )
        before.raise_for_status()
        before_payload = before.json()
        server_state = str(before_payload.get("status") or "")
        if server_state in {"initializing", "shutting_down"}:
            raise RuntimeError(
                f"pooled Docker slot is not resettable yet: status={server_state}"
            )
        binding = before_payload.get("binding") or {}
        bound_session_id = str(binding.get("bound_session_id") or "").strip()
        abort_session_id = bound_session_id or trajectory_id
        response = await self._pool_reset_request(
            slot,
            "POST",
            f"{base_url}/v1/sessions/{abort_session_id}/abort",
            phase="abort_session",
        )
        if response.status_code not in {200, 404}:
            raise RuntimeError(
                "pooled Docker session abort failed: "
                f"status={response.status_code} body={response.text[-1000:]}"
            )
        status = await self._pool_reset_request(
            slot,
            "GET",
            f"{base_url}/v1/status",
            phase="status_after_abort",
        )
        status.raise_for_status()
        status_payload = status.json()
        sessions = status_payload.get("sessions") or {}
        if int(sessions.get("active_count") or 0) != 0 or int(
            sessions.get("desynced_count") or 0
        ) != 0:
            raise RuntimeError(
                f"pooled Docker slot retained open sessions: {sessions}"
            )
        runtime_processes = status_payload.get("runtime_processes") or []
        # BlackboxServer performs the same /proc verification inside its own
        # container.  Keeping it in the status RPC avoids a burst of 24
        # concurrent `docker exec` calls at every rollout wave, which can
        # serialize the Docker daemon long enough to force needless recycling.
        if runtime_processes:
            raise RuntimeError(
                "pooled Docker slot retained Harness/MCP processes: "
                f"{runtime_processes[:8]}"
            )

    async def _pool_reset_request(
        self,
        slot: _DockerPoolSlot,
        method: str,
        url: str,
        *,
        phase: str,
        max_attempts: int = 3,
    ) -> httpx.Response:
        """Run one idempotent reset RPC with a fresh-pool transport retry.

        All three reset operations are idempotent reads or best-effort aborts.
        Retrying them independently avoids replaying a completed slow abort
        merely because the following status response used a stale connection.
        """

        for attempt in range(1, max_attempts + 1):
            try:
                return await slot.http_client.request(
                    method,
                    url,
                    # The whole status/abort/status transaction has its own
                    # (larger) deadline in _release_pooled_slot_limited.  A
                    # shorter per-RPC/connect budget leaves real time for the
                    # idempotent transport retries instead of letting the
                    # first connection consume the entire transaction budget.
                    timeout=httpx.Timeout(
                        self.pool_reset_rpc_timeout_sec,
                        connect=min(3.0, self.pool_reset_rpc_timeout_sec),
                    ),
                )
            except httpx.TransportError as exc:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        "pooled Docker reset transport retry exhausted: "
                        f"phase={phase} attempts={attempt} "
                        f"error={type(exc).__name__}: {exc}"
                    ) from exc
                logger.info(
                    "retrying pooled Docker reset RPC for slot %s after "
                    "transport failure: phase=%s attempt=%d/%d error=%s: %s",
                    slot.container_name,
                    phase,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                )
                await self._renew_pool_http_client(slot)
                await asyncio.sleep(0.1 * attempt)
        raise RuntimeError("unreachable pooled Docker reset retry loop exit")

    async def _renew_pool_http_client(self, slot: _DockerPoolSlot) -> None:
        previous = slot.http_client
        slot.http_client = self._new_pool_http_client()
        # Closing a client whose request was cancelled inside httpcore can
        # itself block on the damaged connection.  Ownership has already moved
        # to the new client, so retire the old one independently and keep the
        # slot release path bounded.
        self._retire_http_client(previous, slot.container_name)
        await asyncio.sleep(0)

    def _retire_http_client(
        self, client: httpx.AsyncClient, container_name: str
    ) -> None:
        task = asyncio.create_task(
            client.aclose(),
            name=f"docker-pool-http-close:{container_name}",
        )
        self._retired_http_close_tasks.add(task)

        def finished(completed: asyncio.Task[Any]) -> None:
            self._retired_http_close_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug(
                    "failed to close retired pooled Docker HTTP client for %s",
                    container_name,
                    exc_info=True,
                )

        task.add_done_callback(finished)

    def _pool_key(self, spec: SandboxSpec, image: str) -> tuple[Any, ...]:
        readonly_mounts = spec.env_args.get("sandbox_readonly_mounts") or []
        return (
            image,
            str(spec.env_args.get("workdir") or "/testbed"),
            self._service_port(spec),
            tuple(str(item) for item in readonly_mounts),
            os.environ.get("HAPT_JOINT_CODE_SNAPSHOT", ""),
        )

    def _pooled_lease(
        self, spec: SandboxSpec, slot: _DockerPoolSlot
    ) -> SandboxLease:
        endpoint = SandboxEndpoint(
            url=f"http://{slot.ip_address}:{slot.service_port}"
        )
        return SandboxLease(
            trajectory_id=spec.trajectory_id,
            provider=self.name,
            sandbox_id=slot.container_name,
            endpoints={
                "blackbox": endpoint,
                **{service.name: SandboxEndpoint(url=endpoint.url) for service in spec.services},
            },
            capabilities={"command", "file", "public_url"},
            metadata={
                "container_name": slot.container_name,
                "image": slot.image,
                "network": self.network,
                "evaluator_only": False,
                "pooled": True,
                "pool_run_id": self.pool_run_label,
                "pool_short_id": self.pool_run_id,
                "pool_uses": slot.uses,
                "ip_address": slot.ip_address,
            },
            raw={"container_name": slot.container_name},
        )

    async def get_public_url(
        self,
        lease: SandboxLease,
        *,
        port: int,
        service_name: str | None = None,
    ) -> SandboxEndpoint:
        if service_name and service_name in lease.endpoints:
            return lease.endpoints[service_name].normalized()
        if "public_url" not in lease.capabilities:
            raise ValueError("network-none Docker evaluator has no public URL")
        ip_address = str(lease.metadata.get("ip_address") or "")
        if not ip_address:
            ip_address = await self._container_ip(str(lease.sandbox_id))
        return SandboxEndpoint(url=f"http://{ip_address}:{port}")

    async def run_command(
        self,
        lease: SandboxLease,
        command: str | list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> CommandResult:
        if lease.sandbox_id is None:
            raise ValueError("Docker lease has no container name")
        docker_command = [self.docker_bin, "exec", "-i"]
        if cwd:
            docker_command.extend(["--workdir", cwd])
        for key, value in sorted((env or {}).items()):
            docker_command.extend(["--env", f"{key}={value}"])
        docker_command.append(lease.sandbox_id)
        effective_stdin = stdin.encode() if isinstance(stdin, str) else stdin
        if isinstance(command, str):
            encoded_command = command.encode("utf-8")
            if len(encoded_command) > _MAX_INLINE_SHELL_BYTES:
                if effective_stdin is not None:
                    raise ValueError(
                        "large Docker shell commands cannot also provide stdin"
                    )
                docker_command.extend(["bash", "-s"])
                effective_stdin = encoded_command
            else:
                docker_command.extend(["bash", "-lc", command])
        else:
            docker_command.extend(command)
        result = await self._run(
            docker_command,
            timeout=timeout,
            stdin=effective_stdin,
        )
        return CommandResult(
            cmd=command,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=result.timed_out,
            metadata={"container_name": lease.sandbox_id},
        )

    async def read_file(
        self,
        lease: SandboxLease,
        path: str,
        *,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        code = (
            "import base64,sys;d=open(sys.argv[1],'rb').read();"
            "n=int(sys.argv[2]);d=d if n<0 else d[:n];"
            "sys.stdout.write(base64.b64encode(d).decode())"
        )
        result = await self.run_command(
            lease,
            ["python3", "-c", code, path, str(-1 if max_bytes is None else max_bytes)],
            timeout=60.0,
        )
        if result.returncode != 0:
            raise OSError(result.stderr)
        data = base64.b64decode(result.stdout)
        return data if encoding is None else data.decode(encoding)

    async def write_file(
        self,
        lease: SandboxLease,
        path: str,
        content: str | bytes,
        *,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict[str, Any]:
        data = content.encode(encoding or "utf-8") if isinstance(content, str) else content
        code = (
            "import os,sys;p=sys.argv[1];"
            "os.makedirs(os.path.dirname(p) or '.',exist_ok=True);"
            "open(p,'ab' if sys.argv[2]=='1' else 'wb').write(sys.stdin.buffer.read())"
        )
        result = await self.run_command(
            lease,
            ["python3", "-c", code, path, "1" if append else "0"],
            timeout=60.0,
            stdin=data,
        )
        if result.returncode != 0:
            raise OSError(result.stderr)
        return {"path": path, "bytes": len(data), "append": append}

    async def write_files(
        self,
        lease: SandboxLease,
        *,
        files: list[dict[str, Any]],
        dist_path: str = "/data",
    ) -> dict[str, Any]:
        written = []
        for item in files:
            relative = str(item.get("path") or item.get("name") or "")
            if not relative:
                raise ValueError("write_files entry requires path or name")
            # Claw rows use audited absolute task mappings such as /root and
            # /tmp/omni_task. Preserve those sandbox paths; conventional
            # relative Dressage artifacts retain the historical /data base.
            target = (
                relative
                if Path(relative).is_absolute()
                else str(Path(dist_path) / relative)
            )
            if item.get("local_path"):
                content = Path(str(item["local_path"])).read_bytes()
            else:
                content = item.get("content", "")
            written.append(
                await self.write_file(lease, target, content)
            )
        return {"files_written": len(written), "files": written}

    def _agent_mount_args(
        self, readonly_mounts: Any = None
    ) -> list[str]:
        python_root = Path(os.path.realpath(os.environ.get("HARNESS_S_CONTROL_PYTHON", __import__("sys").executable))).parents[1]
        node_root = self._node_root()
        node_runtime_root = self._node_runtime_root()
        arguments = [
            "--mount",
            f"type=bind,src={self.project_root},dst=/opt/hapt-runtime,readonly",
            "--mount", f"type=bind,src={self.project_root},dst={self.project_root},readonly",
        ]
        arguments.extend(checkpoint_harness_mount_args(self.project_root))
        # A direct Node runtime is valid for both roots. Docker rejects duplicate
        # mount destinations, unlike the historical separate wrapper/runtime.
        for runtime_root in dict.fromkeys((node_root, node_runtime_root)):
            arguments.extend(["--mount", f"type=bind,src={runtime_root},dst={runtime_root},readonly"])
        if readonly_mounts in (None, ""):
            return arguments + self._snapshot_mount_args()
        if not isinstance(readonly_mounts, list):
            raise ValueError("sandbox_readonly_mounts must be a list of absolute paths")
        seen = {
            str(python_root),
            str(node_root),
            str(node_runtime_root),
            str(self.project_root),
        }
        for raw_path in readonly_mounts:
            path = Path(str(raw_path))
            if not path.is_absolute() or str(path) == "/":
                raise ValueError(
                    f"unsafe sandbox_readonly_mounts entry: {raw_path!r}"
                )
            resolved = path.resolve()
            if not resolved.exists():
                raise ValueError(
                    f"sandbox_readonly_mounts entry does not exist: {resolved}"
                )
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            arguments.extend(
                [
                    "--mount",
                    f"type=bind,src={resolved},dst={resolved},readonly",
                ]
            )
        return arguments + self._snapshot_mount_args()

    def _snapshot_mount_args(self, *, snapshot: str | Path | None = None,
                             run_root: str | Path | None = None) -> list[str]:
        raw = snapshot or os.environ.get("HAPT_JOINT_CODE_SNAPSHOT")
        if not raw:
            return []
        snapshot = Path(raw).resolve()
        run_root = Path(run_root or os.environ.get("HAPT_JOINT_RUN_ROOT", "/invalid")).resolve()
        if snapshot != run_root / "source-snapshot":
            raise ValueError("code snapshot must belong to the current joint run")
        directories = json.loads((snapshot / "mounts.json").read_text())["directories"]
        allowed = {"scripts", "configs/dsh", "configs/training", "third_party/Dressage/dressage",
                   "third_party/Dressage/blackbox_server", "third_party/Dressage/slime/slime",
                   "third_party/Dressage/slime/slime_plugins", "third_party/Dressage/examples/scripts"}
        current = allowed | {"third_party/Dressage/examples/data/swegym"}
        architecture = current | {"third_party/Dressage/slime/scripts/models", "third_party/Dressage/slime/tools"}
        # Both the task sandbox and training container must see the frozen
        # grading converter. Keep historical eight-directory runs readable.
        if (not isinstance(directories, list) or
                len(directories) != len(set(directories)) or
                set(directories) not in (allowed, current, architecture)):
            raise ValueError("unexpected code snapshot mount directories")
        arguments = []
        for relative in directories:
            source = snapshot / relative
            if not source.is_dir() or not source.resolve().is_relative_to(snapshot):
                raise ValueError("missing or unsafe code snapshot directory")
            for base in (self.project_root, Path("/opt/hapt-runtime")):
                arguments += ["--mount", f"type=bind,src={source},dst={base / relative},readonly"]
        sweagent_runtime = os.environ.get("HAPT_SWEAGENT_RUNTIME_ROOT")
        if sweagent_runtime:
            runtime = Path(sweagent_runtime).resolve()
            if runtime != Path("/var/tmp/hapt-sweagent-v101") or not (runtime / "bin/python").is_file():
                raise ValueError("unexpected or missing frozen SWE-agent runtime")
            arguments += ["--mount", f"type=bind,src={runtime},dst={runtime},readonly"]
        return arguments

    def _node_root(self) -> Path:
        return Path(
            os.environ.get("HAPT_NODE_ROOT")
            or self.project_root.parent / "node"
        ).resolve()

    def _node_runtime_root(self) -> Path:
        return Path(
            os.environ.get("HAPT_NODE_RUNTIME_ROOT")
            or self.project_root.parent / "miniconda3" / "envs" / "rllm"
        ).resolve()

    def _service_port(self, spec: SandboxSpec) -> int:
        for service in spec.services:
            if service.name == "blackbox":
                return int(service.port)
        return int(os.environ.get("DRESSAGE_DOCKER_BLACKBOX_PORT", "31000"))

    async def _ensure_image(self, image: str) -> None:
        inspected = await self._run(
            [self.docker_bin, "image", "inspect", image], timeout=60.0
        )
        if inspected.returncode == 0:
            return

        if os.environ.get('HAPT_SWEGYM_IMAGE_PULL_POLICY') == 'never':
            raise FileNotFoundError('offline task image unavailable: ' + image)

        # A rollout group creates several sandboxes for the same task at once,
        # and paired experiments may do so from separate processes.  Serialize
        # cache misses across both levels: concurrent `docker pull` operations
        # for the same large SWE image have previously corrupted the project
        # Docker filesystem.  Acquire the blocking flock in a worker thread so
        # other async provider work can continue while another process pulls.
        lock_root = Path(
            os.environ.get(
                "HAPT_DOCKER_IMAGE_PULL_LOCK_DIR",
                "/tmp/hapt_docker/image-pull-locks",
            )
        )
        lock_path = lock_root / f"{hashlib.sha256(image.encode()).hexdigest()}.lock"

        def acquire_lock():
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX)
            return handle

        lock_handle = await asyncio.to_thread(acquire_lock)
        try:
            # The process that held the lock may have completed the pull.
            inspected = await self._run(
                [self.docker_bin, "image", "inspect", image], timeout=60.0
            )
            if inspected.returncode == 0:
                return
            pulled = await self._run(
                [self.docker_bin, "pull", image], timeout=10800.0
            )
            if pulled.returncode != 0:
                raise RuntimeError(
                    f"failed to pull Docker image {image}: {pulled.stderr[-4000:]}"
                )
        finally:
            await asyncio.to_thread(fcntl.flock, lock_handle, fcntl.LOCK_UN)
            lock_handle.close()

    async def _container_ip(self, container_name: str) -> str:
        result = await self._run(
            [
                self.docker_bin,
                "inspect",
                container_name,
                "--format",
                f"{{{{(index .NetworkSettings.Networks {json.dumps(self.network)}).IPAddress}}}}",
            ],
            timeout=30.0,
        )
        ip_address = result.stdout.strip()
        if result.returncode != 0 or not ip_address:
            raise RuntimeError(
                f"could not resolve Docker sandbox IP for {container_name}: {result.stderr}"
            )
        return ip_address

    async def _wait_for_health(self, base_url: str, *, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        attempts = 0
        last_observation = "no request attempted"
        while asyncio.get_running_loop().time() < deadline:
            attempts += 1
            try:
                response = await self._http_client.get(
                    f"{base_url}/health", timeout=2.0,
                    headers={"Connection": "close"},
                )
                if response.status_code == 200:
                    return
                last_observation = f"HTTP {response.status_code}"
            except httpx.HTTPError as error:
                last_observation = type(error).__name__ + ": " + str(error)
            await asyncio.sleep(0.1)
        raise RuntimeError(
            f"Docker BlackboxServer health timeout: {base_url}; "
            f"attempts={attempts}; last_observation={last_observation}"
        )

    async def _container_startup_diagnostics(self, container_name: str) -> str:
        """Capture the failed server before cleanup removes its evidence."""

        state = await self._run(
            [
                self.docker_bin,
                "inspect",
                "--format",
                "{{json .State}}",
                container_name,
            ],
            timeout=30.0,
        )
        logs = await self._run(
            [self.docker_bin, "logs", "--tail", "100", container_name],
            timeout=30.0,
        )
        state_text = (state.stdout or state.stderr).strip()[-4000:]
        log_text = (logs.stdout + logs.stderr).strip()[-8000:]
        return (
            f"container_state={state_text or '<unavailable>'}; "
            f"container_logs={log_text or '<empty>'}"
        )

    @staticmethod
    def _new_pool_http_client() -> httpx.AsyncClient:
        # One BlackboxServer lives at each pooled slot.  A per-slot client
        # prevents 24 simultaneous release paths from contending for, or
        # invalidating, connections in a cross-host global HTTP pool.
        return httpx.AsyncClient(
            trust_env=False,
            limits=httpx.Limits(
                max_connections=2,
                max_keepalive_connections=2,
                keepalive_expiry=4.0,
            ),
        )

    async def close(self) -> None:
        for slot in tuple(self._pool_slots):
            self._retire_http_client(slot.http_client, slot.container_name)
        self._retire_http_client(self._http_client, "provider-health-client")
        tasks = tuple(self._retired_http_close_tasks)
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=1.0)
        del done
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run(
        self,
        command: list[str],
        *,
        timeout: float | None,
        stdin: bytes | None = None,
    ) -> "_ProcessResult":
        process = await asyncio.create_subprocess_exec(
            *command,
            env={**os.environ, "DOCKER_HOST": self.docker_host},
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return _ProcessResult(
                returncode=None,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                timed_out=True,
            )
        return _ProcessResult(
            returncode=process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=False,
        )

    def _container_name(self, trajectory_id: str, *, evaluator: bool) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", trajectory_id)[:38]
        role = "eval" if evaluator else "agent"
        return f"hapt-{role}-{slug}-{uuid.uuid4().hex[:8]}"

    def _pool_container_name(self) -> str:
        return f"hapt-pool-{self.pool_run_id}-{uuid.uuid4().hex[:8]}"

    def _default_docker_bin(self) -> str:
        fixed = self.project_root.parent / "vab_runtime" / "bin" / "docker"
        return str(fixed if fixed.is_file() else shutil.which("docker") or "docker")


class _ProcessResult:
    def __init__(
        self,
        *,
        returncode: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def _safe_name(value: str, *, max_length: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    safe = safe or "run"
    if len(safe) <= max_length:
        return safe
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
    if max_length <= len(digest):
        return digest[:max_length]
    prefix = safe[: max_length - len(digest) - 1].rstrip(".-")
    return f"{prefix}-{digest}"


def _env_int(name: str, default: int, *, min_value: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_float(name: str, default: float, *, min_value: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value
