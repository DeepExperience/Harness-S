"""Print status for the detached Ray local_bwrap pool."""

from __future__ import annotations

import os

from dressage.config import local_bwrap_manager_name, local_bwrap_namespace


def main() -> None:
    try:
        import ray
    except ImportError as exc:
        raise SystemExit("ray is required to query the local_bwrap pool") from exc

    namespace = local_bwrap_namespace()
    manager_name = local_bwrap_manager_name()
    ray.init(
        address=os.environ.get("DRESSAGE_RAY_ADDRESS", "auto"),
        namespace=namespace,
        ignore_reinit_error=True,
    )
    manager = ray.get_actor(manager_name, namespace=namespace)
    status = ray.get(manager.status.remote())
    _print_status(status)


def _print_status(status: dict) -> None:
    print(
        "TOTAL "
        f"pool_mode={status.get('pool_mode')} "
        f"capacity={status['total_capacity']} "
        f"ready={status['total_ready']} "
        f"leased={status['total_leased']} "
        f"resetting={status.get('total_resetting', 0)} "
        f"restarting={status['total_restarting']} "
        f"failed={status['total_failed']} "
        f"lost={status.get('total_lost', 0)}"
    )
    print()
    print("NODE                       CAP READY LEASED RESET RESTART FAILED LOST DRAIN")
    for node in status["nodes"]:
        label = node["node_ip"] or node["node_id"][:12]
        print(
            f"{label:<26} "
            f"{node['capacity']:>3} "
            f"{node['ready']:>5} "
            f"{node['leased']:>6} "
            f"{node.get('resetting', 0):>5} "
            f"{node['restarting']:>7} "
            f"{node['failed']:>6} "
            f"{node.get('lost', 0):>4} "
            f"{str(node['draining']).lower()}"
        )


if __name__ == "__main__":
    main()
