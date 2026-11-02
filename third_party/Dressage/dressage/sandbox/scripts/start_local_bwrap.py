"""Start the detached Ray local_bwrap pool."""

from __future__ import annotations

import os
import pprint
from typing import Any

from dressage.config import (
    local_bwrap_manager_name,
    local_bwrap_namespace,
    local_bwrap_pool_mode,
    proxy_public_url,
)
from dressage.sandbox.local.bwrap.supervisor import (
    normalize_pool_mode,
)


def main(*, default_pool_mode: str | None = None, force_pool_mode: bool = False) -> None:
    try:
        import ray
    except ImportError as exc:
        raise SystemExit("ray is required to start the local_bwrap pool") from exc

    from dressage.sandbox.local.bwrap import LocalBwrapClusterManager

    pool_mode_value = (
        default_pool_mode
        if force_pool_mode
        else os.environ.get("DRESSAGE_LOCAL_BWRAP_POOL_MODE") or default_pool_mode
    )
    pool_mode = normalize_pool_mode(pool_mode_value or local_bwrap_pool_mode())
    namespace = local_bwrap_namespace()
    ray.init(
        address=os.environ.get("DRESSAGE_RAY_ADDRESS", "auto"),
        namespace=namespace,
        ignore_reinit_error=True,
    )

    manager_name = local_bwrap_manager_name()
    manager = _get_existing_manager(ray, manager_name=manager_name, namespace=namespace)
    if manager is not None and _manager_is_closed(ray, manager):
        ray.kill(manager, no_restart=True)
        manager = None
    if manager is not None:
        current_mode = normalize_pool_mode(_manager_pool_mode(ray, manager))
        if current_mode != pool_mode:
            raise SystemExit(
                "local_bwrap manager already exists with a different pool mode: "
                f"name={manager_name!r} namespace={namespace!r} "
                f"existing={current_mode!r} requested={pool_mode!r}; "
                "stop the current pool before starting another mode"
            )
    if manager is None:
        manager = LocalBwrapClusterManager.options(
            name=manager_name,
            namespace=namespace,
            lifetime="detached",
            get_if_exists=False,
            num_cpus=0.2,
            num_gpus=0,
        ).remote(
            total_servers=_env_int(
                "DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS",
                512,
            ),
            base_port=_env_int(
                "DRESSAGE_LOCAL_BWRAP_BASE_PORT",
                31000,
            ),
            proxy_url=proxy_public_url(),
            namespace=namespace,
            pool_mode=pool_mode,
        )
    ray.get(manager.init_pool.remote())
    status = ray.get(
        manager.wait_ready.remote(
            timeout_s=_env_int(
                "DRESSAGE_LOCAL_BWRAP_WAIT_READY_TIMEOUT_SEC",
                600,
            )
        )
    )
    pprint.pp(status)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_existing_manager(ray: Any, *, manager_name: str, namespace: str) -> Any:
    try:
        return ray.get_actor(manager_name, namespace=namespace)
    except Exception:
        return None


def _manager_is_closed(ray: Any, manager: Any) -> bool:
    try:
        status = ray.get(manager.status.remote())
    except Exception:
        return True
    return bool(status.get("closed"))


def _manager_pool_mode(ray: Any, manager: Any) -> str:
    try:
        return str(ray.get(manager.pool_mode.remote()))
    except Exception:
        status = ray.get(manager.status.remote())
        return str(status.get("pool_mode"))


if __name__ == "__main__":
    main()
