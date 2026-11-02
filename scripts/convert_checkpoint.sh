#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${POLICY_MODEL:?}" "${POLICY_DIST_CHECKPOINT:?}"
ARCHITECTURE="${1:-qwen3.5-4B}"
[[ "${ARCHITECTURE}" == qwen3.5-4B || "${ARCHITECTURE}" == qwen3.5-9B ]] || exit 2
[[ ! -e "${POLICY_DIST_CHECKPOINT}" ]] || { echo 'Destination already exists; choose an empty destination.' >&2; exit 2; }
mkdir -p "$(dirname -- "${POLICY_DIST_CHECKPOINT}")"
exec "${DOCKER_BIN:-docker}" run --rm --gpus "device=${CONVERSION_GPU:-0}" --ipc private --shm-size 32g \
  -v "${ROOT}:${ROOT}:ro" -v "${POLICY_MODEL}:${POLICY_MODEL}:ro" \
  -v "$(dirname -- "${POLICY_DIST_CHECKPOINT}"):$(dirname -- "${POLICY_DIST_CHECKPOINT}")" \
  -e POLICY_MODEL -e POLICY_DIST_CHECKPOINT -e "HARNESS_S_ROOT=${ROOT}" -e "ARCHITECTURE=${ARCHITECTURE}" \
  harness-s:appworld bash -c '
    set -euo pipefail
    cd "${HARNESS_S_ROOT}/third_party/slime"
    source "${HARNESS_S_ROOT}/third_party/model_configs/${ARCHITECTURE}.sh"
    export PYTHONPATH="/root/Megatron-LM:${HARNESS_S_ROOT}/third_party/slime${PYTHONPATH:+:${PYTHONPATH}}"
    work="${POLICY_DIST_CHECKPOINT}.incomplete-$$"
    python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "${POLICY_MODEL}" --save "${work}"
    [[ "$(cat "${work}/latest_checkpointed_iteration.txt")" == release ]]
    mv "${work}" "${POLICY_DIST_CHECKPOINT}"
  '
