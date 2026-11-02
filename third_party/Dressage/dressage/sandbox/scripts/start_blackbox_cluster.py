"""Compatibility alias for starting a blackbox-mode local_bwrap pool."""

from __future__ import annotations

from dressage.sandbox.scripts.start_local_bwrap import main as _start_local_bwrap


def main() -> None:
    _start_local_bwrap(default_pool_mode="blackbox", force_pool_mode=True)


if __name__ == "__main__":
    main()
