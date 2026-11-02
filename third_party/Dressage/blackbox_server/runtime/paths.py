from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from blackbox_server.core.models import utcnow


def make_runtime_id() -> str:
    timestamp = utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:4]
    return f"bbs-{timestamp}-{suffix}"


def ensure_runtime_dir(runtime_root: str, runtime_id: str) -> str:
    runtime_dir = Path(runtime_root) / runtime_id
    runtime_dir.mkdir(parents=True, exist_ok=False)
    return str(runtime_dir)


def remove_runtime_dir(runtime_dir: str | None) -> None:
    if not runtime_dir:
        return
    shutil.rmtree(runtime_dir, ignore_errors=True)
