#!/bin/bash
# Dressage SWE-Gym Qwen3.5-4B Claude Code synchronous four-node experiment.
#
# The task and coding agent are intentionally fixed to SWE-Gym and Claude Code.
# Deployment wrappers may override the model, topology, batch shape, sandbox
# settings, and output paths while keeping machine setup and credentials
# outside this script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../../dressage" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  else
    echo "REPO_ROOT is required when this script is not inside examples/scripts/." >&2
    exit 1
  fi
fi
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
DEFAULT_ENV_FILE="${DEFAULT_ENV_FILE:-${REPO_ROOT}/examples/scripts/default/dressage_env_defaults.sh}"

MODEL_ROOT="${MODEL_ROOT:-}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3.5-4B.sh}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}}"

[[ -f "${MODEL_CONFIG_PATH}" ]] || {
  echo "Missing Slime model config: ${MODEL_CONFIG_PATH}" >&2
  exit 1
}
[[ -f "${DEFAULT_ENV_FILE}" ]] || {
  echo "Missing Dressage default env file: ${DEFAULT_ENV_FILE}" >&2
  exit 1
}
: "${PROMPT_DATA:?PROMPT_DATA must point to Claude Code SWE-Gym Dressage JSONL}"
: "${MODEL_ROOT:?MODEL_ROOT must point to model/checkpoint files, e.g. /path/to/models}"

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
CURRENT_NODE_IP="${CURRENT_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is required and could not be inferred." >&2
  exit 1
fi
export MASTER_ADDR CURRENT_NODE_IP

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONUNBUFFERED=1
PYTHONPATH_ENTRIES=("${REPO_ROOT}" "${SLIME_ROOT}")
if [[ -n "${MEGATRON_ROOT:-}" ]]; then
  PYTHONPATH_ENTRIES=("${MEGATRON_ROOT}" "${PYTHONPATH_ENTRIES[@]}")
fi
PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${WORLD_SIZE:-1}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-${ACTOR_NUM_GPUS_PER_NODE}}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-0}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-0}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-0}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-${CURRENT_NODE_IP}}"
RAY_JOIN_TIMEOUT_SEC="${RAY_JOIN_TIMEOUT_SEC:-600}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-${LOG_DIR:-${REPO_ROOT}/outputs/logs}/ray}"
SOCKET_IFNAME="${SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
HOSTFILE="${HOSTFILE:-}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-3600000}"
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-3600000}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

source "${MODEL_CONFIG_PATH}"
source "${DEFAULT_ENV_FILE}"

# Defaults from the stable SWE-Gym Claude Code run. Keep the experiment
# semantics here; deployment wrappers only need to provide infrastructure.
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.207}"
CLAUDE_CODE_BBS_VERSION="${CLAUDE_CODE_BBS_VERSION:-1.1.0}"
CLAUDE_CODE_PORTABLE_PYTHON_VERSION="${CLAUDE_CODE_PORTABLE_PYTHON_VERSION:-3.10.20}"

TP_SIZE="${TP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
TRAIN_SEED="${TRAIN_SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"

NUM_ROLLOUT="${NUM_ROLLOUT:-}"
NUM_EPOCH="${NUM_EPOCH:-}"
if [[ -z "${NUM_ROLLOUT}" && -z "${NUM_EPOCH}" ]]; then
  NUM_ROLLOUT=500
fi
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"

NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-1}"
USE_TIS="${USE_TIS:-1}"
USE_KL_LOSS="${USE_KL_LOSS:-0}"
KL_COEF="${KL_COEF:-0.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
for toggle_name in NORMALIZE_ADVANTAGES USE_TIS USE_KL_LOSS; do
  toggle_value="${!toggle_name}"
  if [[ "${toggle_value}" != "0" && "${toggle_value}" != "1" ]]; then
    echo "${toggle_name} must be 0 or 1; got ${toggle_value}" >&2
    exit 1
  fi
done
dressage_compute_context_window "${MAX_TOKENS_PER_GPU:-16384}" "${CP_SIZE}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-${CONTEXT_WINDOW}}"
if (( CP_SIZE > 1 && CONTEXT_WINDOW % (CP_SIZE * 2) != 0 )); then
  echo "CONTEXT_WINDOW (${CONTEXT_WINDOW}) must be divisible by 2*CP_SIZE ($((CP_SIZE * 2))) for Megatron zigzag context parallelism." >&2
  exit 1
fi
if (( CONTEXT_WINDOW > MAX_TOKENS_PER_GPU * CP_SIZE )); then
  echo "CONTEXT_WINDOW (${CONTEXT_WINDOW}) exceeds MAX_TOKENS_PER_GPU*CP_SIZE ($((MAX_TOKENS_PER_GPU * CP_SIZE)))." >&2
  exit 1
fi

TOTAL_ACTOR_GPUS="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"
MODEL_PARALLEL_SIZE="$((TP_SIZE * CP_SIZE))"
if (( TOTAL_ACTOR_GPUS % MODEL_PARALLEL_SIZE != 0 )); then
  echo "total actor GPUs (${TOTAL_ACTOR_GPUS}) must be divisible by TP_SIZE*CP_SIZE (${MODEL_PARALLEL_SIZE})" >&2
  exit 1
fi
DP_SIZE="$((TOTAL_ACTOR_GPUS / MODEL_PARALLEL_SIZE))"

COMPACT_RESERVE_TOKENS="${COMPACT_RESERVE_TOKENS:-$((ROLLOUT_MAX_RESPONSE_LEN / 2))}"
DRESSAGE_BLACKBOX_COMPACT_THRESHOLD="${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD:-$((CONTEXT_WINDOW - COMPACT_RESERVE_TOKENS))}"
if [[ "${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}" -le 0 ]]; then
  echo "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD must be positive; got ${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-swegym-${MODEL_NAME,,}-claude-code-grpo-sync}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs/${RUN_NAME}}"
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

PROXY_PUBLIC_HOST="${PROXY_PUBLIC_HOST:-${CURRENT_NODE_IP:-${MASTER_ADDR}}}"
DRESSAGE_SANDBOX_PROVIDER="${DRESSAGE_SANDBOX_PROVIDER:-e2b}"
if dressage_transfer_queue_enabled && [[ -z "${DRESSAGE_TRANSFER_PARAMS:-}" ]]; then
    DRESSAGE_TRANSFER_PARAMS="logprobs"
fi
dressage_apply_common_defaults "${RUN_NAME}" blackbox "${DRESSAGE_SANDBOX_PROVIDER}"
DRESSAGE_PROXY_PUBLIC_URL="${DRESSAGE_PROXY_PUBLIC_URL:-${DRESSAGE_PROXY_URL}}"

if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "e2b" ]]; then
  : "${DRESSAGE_E2B_API_KEY:?DRESSAGE_E2B_API_KEY is required for DRESSAGE_SANDBOX_PROVIDER=e2b}"
fi
if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "custom" ]]; then
  : "${DRESSAGE_PADDOCK_CLASS:?DRESSAGE_PADDOCK_CLASS is required for DRESSAGE_SANDBOX_PROVIDER=custom}"
fi

dressage_validate_proxy_defaults
if [[ "${DRESSAGE_CLEAR_TRAJECTORY_LOGS:-0}" == "1" ]]; then
  dressage_clear_trajectory_logs
fi
dressage_export_common_env

export DRESSAGE_PROXY_PUBLIC_URL
export DRESSAGE_SANDBOX_DEFAULT_IMAGE="${DRESSAGE_SANDBOX_DEFAULT_IMAGE:-}"
export DRESSAGE_E2B_API_KEY="${DRESSAGE_E2B_API_KEY:-}"
export DRESSAGE_E2B_BLACKBOX_PORT="${DRESSAGE_E2B_BLACKBOX_PORT:-31000}"
export DRESSAGE_E2B_TIMEOUT_SEC="${DRESSAGE_E2B_TIMEOUT_SEC:-3600}"
export DRESSAGE_PADDOCK_CLASS="${DRESSAGE_PADDOCK_CLASS:-}"

export DRESSAGE_BLACKBOX_MAX_STEPS="${DRESSAGE_BLACKBOX_MAX_STEPS:-80}"
export DRESSAGE_PROXY_MAX_STEPS_PER_SESSION="${DRESSAGE_PROXY_MAX_STEPS_PER_SESSION:-100}"
export DRESSAGE_ROLLOUT_MAX_RETRIES="${DRESSAGE_ROLLOUT_MAX_RETRIES:-0}"
export DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH="${DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH:-0}"
export DRESSAGE_REWARD_MODULES="${DRESSAGE_REWARD_MODULES:-dressage.recipes.swegym.reward}"
PROXY_MAX_OUTPUT_TOKENS="${PROXY_MAX_OUTPUT_TOKENS:-}"
if [[ -n "${PROXY_MAX_OUTPUT_TOKENS}" ]] && \
   { [[ ! "${PROXY_MAX_OUTPUT_TOKENS}" =~ ^[0-9]+$ ]] || [[ "${PROXY_MAX_OUTPUT_TOKENS}" -le 0 ]]; }; then
  echo "PROXY_MAX_OUTPUT_TOKENS must be a positive integer; got ${PROXY_MAX_OUTPUT_TOKENS}" >&2
  exit 2
fi

echo "effective_parallelism: total_actor_gpus=${TOTAL_ACTOR_GPUS} TP_SIZE=${TP_SIZE} CP_SIZE=${CP_SIZE} DP_SIZE=${DP_SIZE} CONTEXT_WINDOW=${CONTEXT_WINDOW} ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "effective_blackbox: provider=${DRESSAGE_SANDBOX_PROVIDER} proxy=${DRESSAGE_PROXY_URL} max_steps=${DRESSAGE_BLACKBOX_MAX_STEPS} compact_threshold=${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD} max_output_tokens=${PROXY_MAX_OUTPUT_TOKENS:-unbounded}"
echo "effective_claude_code: version=${CLAUDE_CODE_VERSION} bbs_version=${CLAUDE_CODE_BBS_VERSION} portable_python=${CLAUDE_CODE_PORTABLE_PYTHON_VERSION}"
echo "effective_batch: num_rollout=${NUM_ROLLOUT:-auto} num_epoch=${NUM_EPOCH:-none} rollout_batch_size=${ROLLOUT_BATCH_SIZE} n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "effective_randomness: train_seed=${TRAIN_SEED} rollout_seed=${ROLLOUT_SEED} temperature=${ROLLOUT_TEMPERATURE:-1.0}"
echo "effective_grpo: normalize_advantages=${NORMALIZE_ADVANTAGES} use_tis=${USE_TIS} use_kl_loss=${USE_KL_LOSS} kl_coef=${KL_COEF} entropy_coef=${ENTROPY_COEF} eps_clip=${EPS_CLIP} eps_clip_high=${EPS_CLIP_HIGH} kl_loss_coef=${KL_LOSS_COEF} kl_loss_type=${KL_LOSS_TYPE}"

TQ_PROXY_ARGS=()
TQ_TRAINING_ARGS=()
TRAIN_ENTRY=(python3 "${HAPT_TRAIN_ENTRY:-${SLIME_ROOT}/train.py}")
if dressage_transfer_queue_enabled; then
    NORMALIZED_TRANSFER_PARAMS="${DRESSAGE_TRANSFER_PARAMS//,/ }"
    read -r -a TRANSFER_PARAMS <<< "${NORMALIZED_TRANSFER_PARAMS}"
    TQ_PROXY_ARGS=(
        --enable-transfer-queue
        --transfer-queue-config "${DRESSAGE_TRANSFER_QUEUE_CONFIG}"
        --transfer-queue-retention-seconds "${DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS}"
        --transfer-params "${TRANSFER_PARAMS[@]}"
    )
    if [[ " ${NORMALIZED_TRANSFER_PARAMS} " == *" routed_experts "* ]]; then
        TQ_PROXY_ARGS+=(--use-rollout-routing-replay)
        TQ_TRAINING_ARGS+=(--use-rollout-routing-replay)
    fi
    TRAIN_ENTRY=(python3 -m dressage.training.tq_train)
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
PROXY_ARGS=(
  --tokenizer-path "${TOKENIZER_PATH}"
  --host "${PROXY_HOST}"
  --port "${PROXY_PORT}"
  --token-build-mode "${TOKEN_BUILD_MODE}"
  --token-build-model "${TOKEN_BUILD_MODEL}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --context-window "${CONTEXT_WINDOW}"
  "${TQ_PROXY_ARGS[@]}"
)
if [[ -n "${PROXY_MAX_OUTPUT_TOKENS}" ]]; then
  PROXY_ARGS+=(--max-output-tokens "${PROXY_MAX_OUTPUT_TOKENS}")
fi
if [[ "${RECORD_TOKEN_VERSIONS:-1}" == "1" ]]; then
  PROXY_ARGS+=(--record-token-versions)
elif [[ "${RECORD_TOKEN_VERSIONS}" != "0" ]]; then
  echo "RECORD_TOKEN_VERSIONS must be 0 or 1; got ${RECORD_TOKEN_VERSIONS}" >&2
  exit 1
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist/}"
CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_ROOT}/"
  --save-interval "${SAVE_INTERVAL:-10}"
)
if [[ "${SAVE_OPTIMIZER:-1}" == "0" ]]; then
  CKPT_ARGS+=(--no-save-optim)
elif [[ "${SAVE_OPTIMIZER}" != "1" ]]; then
  echo "SAVE_OPTIMIZER must be 0 or 1; got ${SAVE_OPTIMIZER}" >&2
  exit 1
else
  # CPU-offloaded precision-aware Adam produces rank-local Python lists in
  # Megatron's default dp_reshardable format.  Those lists are not reliably
  # reloadable even with the same topology.  Gather a model-space optimizer
  # checkpoint at validation boundaries so Policy+Adam restarts are real.
  CKPT_ARGS+=(--dist-ckpt-optim-fully-reshardable)
fi
if [[ "${RESUME_FROM_CKPT:-0}" == "1" ]]; then
  : "${RESUME_CKPT_ROOT:?RESUME_CKPT_ROOT is required when RESUME_FROM_CKPT=1}"
  CKPT_ARGS+=(--load "${RESUME_CKPT_ROOT%/}/")
  if [[ -n "${RESUME_CKPT_STEP:-}" ]]; then
    CKPT_ARGS+=(--ckpt-step "${RESUME_CKPT_STEP}")
  fi
  if [[ "${RESUME_LOAD_OPTIM:-1}" != "1" ]]; then
    CKPT_ARGS+=(--no-load-optim)
  fi
else
  CKPT_ARGS+=(--no-load-optim)
fi

ROLLOUT_ARGS=(
  --rollout-function-path harness_s.backend.generate_rollout
  --custom-generate-function-path "${DRESSAGE_CUSTOM_GENERATE_FUNCTION_PATH:-dressage.rollout.generate.blackbox_dispatch_swegym.generate}"
  --custom-rm-path dressage.reward.custom_rm.custom_rm
  --data-source-path dressage.rollout.data_source.DressageDataSource
  --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
  --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
  --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data
  --eval-function-path harness_s.backend.evaluate
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)
if [[ -n "${EVAL_CONFIG:-}" ]]; then
  ROLLOUT_ARGS+=(
    --eval-function-path "${EVAL_FUNCTION_PATH:-dressage.rollout.appworld_eval.generate_rollout}"
    --eval-config "${EVAL_CONFIG}"
    --eval-interval "${EVAL_INTERVAL:-21}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
    --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN:-${ROLLOUT_MAX_CONTEXT_LEN}}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --custom-eval-rollout-log-function-path "${CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH:-dressage.rollout.appworld_eval.log_eval_rollout_data}"
  )
fi
if [[ -n "${NUM_ROLLOUT}" ]]; then
  ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
else
  ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
fi
if [[ "${ROLLOUT_SHUFFLE:-1}" == "1" ]]; then
  ROLLOUT_ARGS+=(--rollout-shuffle)
elif [[ "${ROLLOUT_SHUFFLE}" != "0" ]]; then
  echo "ROLLOUT_SHUFFLE must be 0 or 1; got ${ROLLOUT_SHUFFLE}" >&2
  exit 1
fi

PERF_ARGS=(
  --seq-length "${CONTEXT_WINDOW}"
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-coef "${KL_COEF}"
  --kl-loss-coef "${KL_LOSS_COEF}"
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
)
if [[ "${NORMALIZE_ADVANTAGES}" == "1" ]]; then
  GRPO_ARGS+=(--normalize-advantages)
fi
if [[ "${USE_TIS}" == "1" ]]; then
  # Slime's vanilla token-level TIS uses its established [0, 2] defaults.
  # Keep rollout-logprob bypass and custom MIS/RS opt-in and out of this recipe.
  GRPO_ARGS+=(--use-tis)
fi
if [[ "${USE_KL_LOSS}" == "1" ]]; then
  GRPO_ARGS+=(
    --use-kl-loss
    --kl-loss-type "${KL_LOSS_TYPE}"
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${ADAM_BETA1:-0.9}"
  --adam-beta2 "${ADAM_BETA2:-0.98}"
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-dressage-swegym}"
    --wandb-group "${WANDB_GROUP:-${RUN_NAME}}"
  )
  if [[ -n "${WANDB_TEAM:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  # Credentials normally travel in Ray's job-scoped runtime environment.
  # Keep them out of train.py argv and Ray job logs unless explicitly needed
  # for a legacy deployment.
  if [[ "${WANDB_KEY_IN_COMMAND_LINE:-0}" == "1" && -n "${WANDB_KEY:-${WANDB_API_KEY:-}}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY:-${WANDB_API_KEY}}")
  fi
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
  fi
  if [[ "${WANDB_DISABLE_RANDOM_SUFFIX:-0}" == "1" || -n "${WANDB_RUN_ID:-}" ]]; then
    WANDB_ARGS+=(--disable-wandb-random-suffix)
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-4}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.5}"
  --sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder
  --sglang-log-level warning
  --sglang-router-port "${SGLANG_ROUTER_PORT}"
  --router-policy consistent_hashing
)
if [[ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
elif [[ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}" != "0" ]]; then
  echo "SGLANG_ENABLE_DETERMINISTIC_INFERENCE must be 0 or 1" >&2
  exit 1
fi

MISC_ARGS=(
  --seed "${TRAIN_SEED}"
  --custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --loss-mask-type "${LOSS_MASK_TYPE:-qwen3_5}"
)

DEBUG_ARGS=()
if [[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]]; then
  DEBUG_ROLLOUT_PATH_DEFAULT="${SAVE_ROOT}/rollout_data/{rollout_id}.pt"
  DEBUG_ARGS=(
    --debug-rollout-only
    --save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA:-${DEBUG_ROLLOUT_PATH_DEFAULT}}"
  )
fi

PROXY_LOG_FILE="${PROXY_LOG_FILE:-${LOG_DIR}/dressage-proxy.log}"
PROXY_PID_FILE="${PROXY_PID_FILE:-${LOG_DIR}/dressage-proxy.pid}"
RAY_HEAD_PID=""
python3 -m dressage.proxy.server "${PROXY_ARGS[@]}" >"${PROXY_LOG_FILE}" 2>&1 &
echo $! >"${PROXY_PID_FILE}"

cleanup() {
  status=$?
  set +e
  trap - EXIT INT TERM
  # The Ray head belongs to this per-arm container.  Kill it by the exact PID
  # captured below instead of using `ray stop`, which could target an
  # unrelated cluster if the environment were ever misconfigured.
  if [[ -n "${RAY_HEAD_PID}" ]] && kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    kill -TERM "${RAY_HEAD_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${RAY_HEAD_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
      kill -KILL "${RAY_HEAD_PID}" 2>/dev/null || true
    fi
    wait "${RAY_HEAD_PID}" 2>/dev/null || true
  fi
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    kill "$(cat "${PROXY_PID_FILE}")" 2>/dev/null || true
    rm -f "${PROXY_PID_FILE}"
  fi
  if [[ "${DRESSAGE_DOCKER_POOL_SIZE:-0}" -gt 0 && -n "${DRESSAGE_DOCKER_POOL_RUN_ID:-}" && -x "${HAPT_DOCKER_BIN:-}" ]]; then
    mapfile -t pool_container_ids < <(
      "${HAPT_DOCKER_BIN}" --host "${DOCKER_HOST}" ps -aq \
        --filter "label=hapt.dressage.pool_run=${DRESSAGE_DOCKER_POOL_RUN_ID}"
    )
    pool_cleanup_pids=()
    for pool_container_id in "${pool_container_ids[@]}"; do
      if [[ "${pool_container_id}" =~ ^[0-9a-f]{12,64}$ ]]; then
        "${HAPT_DOCKER_BIN}" --host "${DOCKER_HOST}" rm --force "${pool_container_id}" &
        pool_cleanup_pids+=("$!")
        if [[ "${#pool_cleanup_pids[@]}" -ge 8 ]]; then
          for pool_cleanup_pid in "${pool_cleanup_pids[@]}"; do
            wait "${pool_cleanup_pid}" || true
          done
          pool_cleanup_pids=()
        fi
      fi
    done
    for pool_cleanup_pid in "${pool_cleanup_pids[@]}"; do
      wait "${pool_cleanup_pid}" || true
    done
  fi
    exit "${status}"
}
trap cleanup EXIT

if [[ "${DRESSAGE_SKIP_PROXY_HEALTHCHECK:-0}" != "1" ]]; then
  DRESSAGE_PROXY_START_TIMEOUT_SEC="${DRESSAGE_PROXY_START_TIMEOUT_SEC:-60}"
  if ! [[ "${DRESSAGE_PROXY_START_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DRESSAGE_PROXY_START_TIMEOUT_SEC must be a positive integer; got ${DRESSAGE_PROXY_START_TIMEOUT_SEC}" >&2
    exit 1
  fi
  for i in $(seq 1 "${DRESSAGE_PROXY_START_TIMEOUT_SEC}"); do
    curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1 && break
    if [[ "${i}" -eq "${DRESSAGE_PROXY_START_TIMEOUT_SEC}" ]]; then
      echo "Dressage proxy failed; see ${PROXY_LOG_FILE}" >&2
      exit 1
    fi
    sleep 1
  done
fi

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST:-}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_ROOT}"
mkdir -p "${RAY_TEMP_DIR}"
RAY_START_EXTRA_ARGS=()
for port_spec in \
  "node-manager-port:${RAY_NODE_MANAGER_PORT:-}" \
  "object-manager-port:${RAY_OBJECT_MANAGER_PORT:-}" \
  "metrics-export-port:${RAY_METRICS_EXPORT_PORT:-}" \
  "min-worker-port:${RAY_MIN_WORKER_PORT:-}" \
  "max-worker-port:${RAY_MAX_WORKER_PORT:-}"; do
  option="${port_spec%%:*}"
  value="${port_spec#*:}"
  if [[ -n "${value}" ]]; then
    RAY_START_EXTRA_ARGS+=("--${option}=${value}")
  fi
done
ray start --head --block --port="${RAY_PORT}" --node-ip-address="${RAY_NODE_IP_ADDRESS}" \
  --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats \
  --dashboard-host="${RAY_DASHBOARD_HOST}" --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
  "${RAY_START_EXTRA_ARGS[@]}" \
  --temp-dir="${RAY_TEMP_DIR}" &
RAY_HEAD_PID="$!"
sleep 10


JOIN_DEADLINE=$((SECONDS + RAY_JOIN_TIMEOUT_SEC))
while true; do
  NODE_COUNT="$(ray status 2>/dev/null | grep -c node_ || true)"
  echo "Current Ray node count: ${NODE_COUNT}; expected ACTOR_NUM_NODES: ${ACTOR_NUM_NODES}"
  [[ "${NODE_COUNT}" -eq "${ACTOR_NUM_NODES}" ]] && break
  if [[ "${SECONDS}" -ge "${JOIN_DEADLINE}" ]]; then
    echo "Timed out waiting for Ray workers to join." >&2
    exit 1
  fi
  sleep 10
done

RUNTIME_ENV_JSON="$(
  python3 - <<'PY'
import json
import os

# The key-driven construction below forwards the same mappings as conventional
# runtime_env JSON, including:
# "DRESSAGE_BLACKBOX_MAX_STEPS": "${DRESSAGE_BLACKBOX_MAX_STEPS}"
# "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD": "${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}"
keys = [
    "no_proxy",
    "NO_PROXY",
    "MASTER_ADDR",
    "PYTHONPATH",
    "TMPDIR",
    "RAY_TMPDIR",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "WANDB_KEY",
    "WANDB_API_KEY",
    "WANDB_MODE",
    "DRESSAGE_PROXY_URL",
    "DRESSAGE_PROXY_PUBLIC_URL",
    "DRESSAGE_PADDOCK_MODE",
    "DRESSAGE_PADDOCK_CLASS",
    "DRESSAGE_SANDBOX_PROVIDER",
    "DRESSAGE_SANDBOX_DEFAULT_IMAGE",
    "DRESSAGE_E2B_API_KEY",
    "DRESSAGE_E2B_BLACKBOX_PORT",
    "DRESSAGE_E2B_TIMEOUT_SEC",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR",
    "DRESSAGE_REWARD_MODULES",
    "DRESSAGE_BLACKBOX_MAX_STEPS",
    "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD",
    "DRESSAGE_PROXY_MAX_STEPS_PER_SESSION",
    "DRESSAGE_ROLLOUT_MAX_RETRIES",
    "DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH",
    "DRESSAGE_ENABLE_TRANSFER_QUEUE",
    "DRESSAGE_TRANSFER_QUEUE_CONFIG",
    "DRESSAGE_TRANSFER_PARAMS",
    "DRESSAGE_TRANSFER_QUEUE_STORE_ID",
]
for raw_key in os.environ.get("DRESSAGE_EXTRA_RUNTIME_ENV_KEYS", "").split(","):
    key = raw_key.strip()
    if key and key not in keys:
        keys.append(key)
env = {}
for key in keys:
    value = os.environ.get(key)
    if value not in (None, ""):
        env[key] = value
print(json.dumps({"env_vars": env}))
PY
)"

RAY_JOB_COMMAND=(ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_ENTRY[@]}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  "${DEBUG_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${TQ_TRAINING_ARGS[@]}")

# The dashboard can report the node before its per-node job agent has
# registered. Retry only that exact transient; never relaunch a real failed
# training job.
RAY_JOB_SUBMIT_RETRIES="${RAY_JOB_SUBMIT_RETRIES:-12}"
RAY_JOB_SUBMIT_RETRY_SEC="${RAY_JOB_SUBMIT_RETRY_SEC:-5}"
RAY_JOB_SUBMIT_LOG="${LOG_DIR}/ray-job-submit.log"
for attempt in $(seq 1 "${RAY_JOB_SUBMIT_RETRIES}"); do
  set +e
  "${RAY_JOB_COMMAND[@]}" 2>&1 | tee "${RAY_JOB_SUBMIT_LOG}"
  submit_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${submit_status}" -eq 0 ]]; then
    exit 0
  fi
  if ! grep -q "No available agent to submit job" "${RAY_JOB_SUBMIT_LOG}"; then
    exit "${submit_status}"
  fi
  if [[ "${attempt}" -eq "${RAY_JOB_SUBMIT_RETRIES}" ]]; then
    exit "${submit_status}"
  fi
  echo "Ray job agent is not ready (attempt ${attempt}/${RAY_JOB_SUBMIT_RETRIES}); retrying in ${RAY_JOB_SUBMIT_RETRY_SEC}s."
  sleep "${RAY_JOB_SUBMIT_RETRY_SEC}"
done
