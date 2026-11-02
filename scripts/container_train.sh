#!/usr/bin/env bash
set -euo pipefail
: "${HARNESS_S_ROOT:?}" "${HARNESS_S_RUN:?}"
cd "${HARNESS_S_ROOT}"
unset RAY_ADDRESS RAY_NAMESPACE HOSTFILE
export PYTHONPATH="${HARNESS_S_ROOT}/src:${HARNESS_S_ROOT}/third_party/Dressage:${HARNESS_S_ROOT}/third_party/slime:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
# Only this local module emits shell-quoted assignments. It never prints API keys.
eval "$(python -m harness_s.runtime_env)"
exec bash third_party/Dressage/examples/scripts/train_appworld.sh
