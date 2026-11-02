#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${DOCKER_BIN:-docker}" build -f "${ROOT}/docker/Dockerfile" -t harness-s:appworld "${ROOT}"
