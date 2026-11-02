#!/bin/bash
# Build the two immutable artifacts used to bootstrap Claude Code in the
# older Python task images used by coding benchmarks. Upload the resulting
# files to storage reachable from the sandbox and pass their URLs to the
# SWE-Gym data preparation script.

set -euo pipefail

CLAUDE_CODE_ARTIFACT_DIR="${CLAUDE_CODE_ARTIFACT_DIR:-/tmp/dressage/claude_code_artifacts}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.207}"
CLAUDE_CODE_BINARY_SHA256="${CLAUDE_CODE_BINARY_SHA256:-85e7e988a392d859f90802ca21fb26e89d3c9ab527f5ed0b08df3955e34d5c83}"
CLAUDE_CODE_DIST_URL="${CLAUDE_CODE_DIST_URL:-https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}/linux-x64/claude}"
CLAUDE_CODE_BBS_VERSION="${CLAUDE_CODE_BBS_VERSION:-custom}"
: "${CLAUDE_CODE_BBS_WHEEL_URL:?Set CLAUDE_CODE_BBS_WHEEL_URL to a Blackbox Server wheel URL or path}"
CLAUDE_CODE_PORTABLE_PYTHON_VERSION="${CLAUDE_CODE_PORTABLE_PYTHON_VERSION:-3.10.20}"

CLAUDE_CODE_BINARY_FILE="${CLAUDE_CODE_ARTIFACT_DIR}/claude-${CLAUDE_CODE_VERSION}-linux-x64"
CLAUDE_CODE_RUNTIME_FILE="${CLAUDE_CODE_ARTIFACT_DIR}/claude-code-runtime-python-${CLAUDE_CODE_PORTABLE_PYTHON_VERSION}-bbs-${CLAUDE_CODE_BBS_VERSION}.tar.gz"

mkdir -p "${CLAUDE_CODE_ARTIFACT_DIR}"

binary_is_valid=0
if [[ -s "${CLAUDE_CODE_BINARY_FILE}" ]]; then
  actual_sha256="$(sha256sum "${CLAUDE_CODE_BINARY_FILE}" | awk '{print $1}')"
  if [[ "${actual_sha256}" == "${CLAUDE_CODE_BINARY_SHA256}" ]]; then
    binary_is_valid=1
  else
    echo "Discarding Claude Code binary with unexpected sha256=${actual_sha256}" >&2
    rm -f "${CLAUDE_CODE_BINARY_FILE}"
  fi
fi

if [[ "${binary_is_valid}" != "1" ]]; then
  binary_tmp="${CLAUDE_CODE_BINARY_FILE}.tmp.$$"
  trap 'rm -f "${binary_tmp:-}"' EXIT
  curl --connect-timeout 15 --max-time 300 --retry 3 --retry-delay 2 \
    -fsSL "${CLAUDE_CODE_DIST_URL}" -o "${binary_tmp}"
  echo "${CLAUDE_CODE_BINARY_SHA256}  ${binary_tmp}" | sha256sum -c -
  chmod 0755 "${binary_tmp}"
  mv -f "${binary_tmp}" "${CLAUDE_CODE_BINARY_FILE}"
  trap - EXIT
fi

if [[ ! -s "${CLAUDE_CODE_RUNTIME_FILE}" || "${CLAUDE_CODE_REBUILD_RUNTIME:-0}" == "1" ]]; then
  command -v uv >/dev/null 2>&1 || {
    echo "uv is required to prepare the portable Claude Code sandbox runtime" >&2
    exit 1
  }
  build_root="$(mktemp -d "${CLAUDE_CODE_ARTIFACT_DIR}/.cc-runtime.XXXXXX")"
  runtime_tmp="${CLAUDE_CODE_RUNTIME_FILE}.tmp.$$"
  cleanup_build() {
    rm -rf "${build_root}"
    rm -f "${runtime_tmp}"
  }
  trap cleanup_build EXIT

  UV_PYTHON_INSTALL_DIR="${build_root}/python-install" \
    uv python install "${CLAUDE_CODE_PORTABLE_PYTHON_VERSION}"
  portable_python_root="$(find "${build_root}/python-install" -mindepth 1 -maxdepth 1 -type d -name "cpython-${CLAUDE_CODE_PORTABLE_PYTHON_VERSION}-linux-x86_64-gnu" -print -quit)"
  [[ -n "${portable_python_root}" ]] || {
    echo "uv did not materialize the expected portable Python ${CLAUDE_CODE_PORTABLE_PYTHON_VERSION}" >&2
    exit 1
  }

  mkdir -p "${build_root}/cc-runtime/python" "${build_root}/cc-runtime/bbs-site"
  cp -a "${portable_python_root}/." "${build_root}/cc-runtime/python/"
  "${build_root}/cc-runtime/python/bin/python3" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --only-binary=:all: \
    --target "${build_root}/cc-runtime/bbs-site" \
    "${CLAUDE_CODE_BBS_WHEEL_URL}"

  PYTHONPATH="${build_root}/cc-runtime/bbs-site" \
    "${build_root}/cc-runtime/python/bin/python3" -c \
    'import blackbox_server, fastapi, pydantic_core, uvicorn'
  tar -C "${build_root}" -czf "${runtime_tmp}" cc-runtime
  mv -f "${runtime_tmp}" "${CLAUDE_CODE_RUNTIME_FILE}"
  cleanup_build
  trap - EXIT
fi

echo "claude_code_binary_file=${CLAUDE_CODE_BINARY_FILE}"
echo "claude_code_binary_sha256=${CLAUDE_CODE_BINARY_SHA256}"
echo "claude_code_runtime_file=${CLAUDE_CODE_RUNTIME_FILE}"
