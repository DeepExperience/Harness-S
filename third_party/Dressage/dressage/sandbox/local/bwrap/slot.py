"""Local blackbox server slot metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable
import uuid


SLOT_EMPTY = "EMPTY"
SLOT_STARTING = "STARTING"
SLOT_READY = "READY"
SLOT_LEASED = "LEASED"
SLOT_RELEASING = "RELEASING"
SLOT_RESETTING = "RESETTING"
SLOT_RESTARTING = "RESTARTING"
SLOT_FAILED = "FAILED"
SLOT_DEAD = "DEAD"
SLOT_LOST = "LOST"

_ARCHIVABLE_DIRS = ("home", "work", "runtime", "tmp")
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_session_path_component(value: str | None) -> str:
    """Return a filesystem-safe, non-empty session path component."""
    if not value:
        return "unknown-session"
    safe = _SAFE_SESSION_RE.sub("_", value).strip("._-")
    return safe or "unknown-session"


@dataclass(slots=True)
class SlotConfig:
    slot_id: int
    port: int
    bind_host: str
    advertise_host: str
    base_dir: Path
    blackbox_type: str = "opencode"
    memory_high_bytes: int = 1536 * 1024 * 1024
    memory_max_bytes: int = 2 * 1024 * 1024 * 1024
    pids_max: int = 128
    nofile: int = 512

    @property
    def sandbox_url(self) -> str:
        return f"http://{self.advertise_host}:{self.port}"

    @property
    def slot_dir(self) -> Path:
        return self.base_dir / f"{self.slot_id:04d}"

    @property
    def home_dir(self) -> Path:
        return self.slot_dir / "home"

    @property
    def work_dir(self) -> Path:
        return self.slot_dir / "work"

    @property
    def runtime_dir(self) -> Path:
        return self.slot_dir / "runtime"

    @property
    def tmp_dir(self) -> Path:
        return self.slot_dir / "tmp"

    @property
    def log_dir(self) -> Path:
        return self.slot_dir / "logs"

    @property
    def archive_dir(self) -> Path:
        return self.slot_dir / "archives"

    @property
    def active_dirs(self) -> dict[str, Path]:
        return {
            "home": self.home_dir,
            "work": self.work_dir,
            "runtime": self.runtime_dir,
            "tmp": self.tmp_dir,
        }

    def ensure_dirs(self) -> None:
        for path in (
            self.home_dir,
            self.work_dir,
            self.runtime_dir,
            self.tmp_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def clear_runtime_dirs(self) -> None:
        self.reset_runtime_dirs(preserve_artifacts=False)

    def reset_runtime_dirs(
        self,
        *,
        preserve_artifacts: bool = False,
        session_id: str | None = None,
        lease_id: str | None = None,
        generation: int | None = None,
        reason: str | None = None,
        archive_dirs: Iterable[str] = _ARCHIVABLE_DIRS,
        archive_max_per_slot: int | None = None,
        archive_ttl_sec: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Clear active slot dirs, optionally rotating them into an archive.

        When preservation is enabled, selected active dirs are moved to
        ``archives/<session_id>/`` and fresh active dirs are created. New sessions
        never see archived contents because ``archives`` is not bind-mounted into
        the sandbox.
        """
        selected = {name for name in archive_dirs if name in self.active_dirs}
        archive_path: Path | None = None
        if preserve_artifacts and selected:
            archive_path = self._allocate_archive_dir(session_id)
            archive_path.mkdir(parents=True, exist_ok=False)
            payload = {
                "session_id": session_id,
                "bound_session_id": session_id,
                "lease_id": lease_id,
                "generation": generation,
                "slot_id": self.slot_id,
                "port": self.port,
                "blackbox_type": self.blackbox_type,
                "reason": reason,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "archive_dirs": sorted(selected),
            }
            if metadata:
                payload.update(metadata)
            (archive_path / "metadata.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        for name, path in self.active_dirs.items():
            if path.exists():
                if archive_path is not None and name in selected:
                    path.rename(archive_path / name)
                else:
                    shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if archive_path is not None:
            self.prune_archives(
                max_per_slot=archive_max_per_slot,
                ttl_sec=archive_ttl_sec,
            )
        return archive_path

    def prune_archives(
        self,
        *,
        max_per_slot: int | None = None,
        ttl_sec: float | None = None,
    ) -> None:
        if not self.archive_dir.exists():
            return
        archives = [path for path in self.archive_dir.iterdir() if path.is_dir()]
        now = time.time()
        if ttl_sec is not None and ttl_sec > 0:
            for path in archives:
                try:
                    if now - path.stat().st_mtime > ttl_sec:
                        shutil.rmtree(path)
                except FileNotFoundError:
                    pass
        if max_per_slot is not None and max_per_slot >= 0:
            archives = [path for path in self.archive_dir.iterdir() if path.is_dir()]
            archives.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            for path in archives[max_per_slot:]:
                shutil.rmtree(path, ignore_errors=True)

    def _allocate_archive_dir(self, session_id: str | None) -> Path:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        safe_session_id = safe_session_path_component(session_id)
        candidate = self.archive_dir / safe_session_id
        if not candidate.exists():
            return candidate
        # The Dressage rollout path generates unique bound_session_id values.
        # Keep this fallback only for manual/debug/replay collisions; generation
        # stays in metadata and is not part of the default archive directory name.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = 1
        while True:
            candidate = self.archive_dir / f"{safe_session_id}-collision-{timestamp}-{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "port": self.port,
            "bind_host": self.bind_host,
            "advertise_host": self.advertise_host,
            "sandbox_url": self.sandbox_url,
            "base_dir": str(self.base_dir),
            "blackbox_type": self.blackbox_type,
            "memory_high_bytes": self.memory_high_bytes,
            "memory_max_bytes": self.memory_max_bytes,
            "pids_max": self.pids_max,
            "nofile": self.nofile,
        }


@dataclass(slots=True)
class SlotRuntime:
    config: SlotConfig
    generation: int = 0
    status: str = SLOT_EMPTY
    process_pid: int | None = None
    process: Any | None = None
    cleanup_token: str | None = None
    supervisor_run_id: str | None = None
    lease_id: str | None = None
    trajectory_id: str | None = None
    acquired_ts: float | None = None
    last_health_ts: float = 0.0
    last_error: str | None = None

    @property
    def sandbox_url(self) -> str:
        return self.config.sandbox_url

    @property
    def is_available(self) -> bool:
        return self.status == SLOT_READY and self.lease_id is None

    def rotate_cleanup_token(self, *, supervisor_run_id: str | None = None) -> str:
        """Assign a fresh opaque cleanup token for this slot generation."""
        self.supervisor_run_id = supervisor_run_id
        self.cleanup_token = uuid.uuid4().hex
        return self.cleanup_token

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "generation": self.generation,
            "status": self.status,
            "process_pid": self.process_pid,
            "cleanup_token": self.cleanup_token,
            "supervisor_run_id": self.supervisor_run_id,
            "lease_id": self.lease_id,
            "trajectory_id": self.trajectory_id,
            "acquired_ts": self.acquired_ts,
            "last_health_ts": self.last_health_ts,
            "last_error": self.last_error,
        }
