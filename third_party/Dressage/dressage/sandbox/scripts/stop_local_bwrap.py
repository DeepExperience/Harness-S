"""Stop the detached Ray local_bwrap pool."""

from __future__ import annotations

import os
import pprint

from dressage.config import local_bwrap_manager_name, local_bwrap_namespace


def main() -> None:
    try:
        import ray
    except ImportError:
        print("ray is not installed; no local_bwrap pool to stop")
        return

    namespace = local_bwrap_namespace()
    manager_name = local_bwrap_manager_name()
    destroy_actors = _env_bool("DRESSAGE_LOCAL_BWRAP_DESTROY_ACTORS_ON_STOP", True)

    try:
        ray.init(
            address=os.environ.get("DRESSAGE_RAY_ADDRESS", "auto"),
            namespace=namespace,
            ignore_reinit_error=True,
        )
    except Exception as exc:
        print(f"ray is not running or not reachable; nothing to stop: {_summary(exc)}")
        return

    try:
        manager = ray.get_actor(manager_name, namespace=namespace)
    except Exception:
        print(
            "local_bwrap manager not found; nothing to stop "
            f"name={manager_name!r} namespace={namespace!r}"
        )
        return

    try:
        result = ray.get(manager.shutdown.remote(destroy_supervisors=destroy_actors))
    except Exception as exc:
        result = {"stopped": False, "error": _summary(exc)}
    pprint.pp(result)

    if destroy_actors:
        try:
            ray.kill(manager, no_restart=True)
        except Exception as exc:
            print(f"failed to destroy local_bwrap manager actor: {_summary(exc)}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


if __name__ == "__main__":
    main()
