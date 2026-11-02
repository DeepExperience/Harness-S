"""Atomic critical state and best-effort optional telemetry."""
import hashlib
import json
import logging
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry(path, value):
    try:
        atomic_json(path, value)
    except (OSError, ValueError) as exc:
        logging.warning("Optional telemetry unavailable: %s", exc)
