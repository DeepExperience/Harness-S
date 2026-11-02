"""Translate a frozen recipe into the established backend launch settings."""
import os
from pathlib import Path
import shlex
from .io import read_json
from .config import api_key_envs


def settings():
    root = Path(os.environ['HARNESS_S_ROOT'])
    run = Path(os.environ['HARNESS_S_RUN'])
    c = read_json(run/'config.json')
    r,s,o = c['runtime'],c['sampling'],c['optimizer']
    env = dict(REPO_ROOT=root/'third_party/Dressage',SLIME_ROOT=root/'third_party/slime',
        MEGATRON_ROOT='/root/Megatron-LM',MODEL_ROOT=Path(c['policy']['hf_checkpoint']).parent,
        MODEL_NAME=Path(c['policy']['hf_checkpoint']).name,
        MODEL_CONFIG_PATH=root/f'third_party/model_configs/{c["policy"]["architecture"]}.sh',
        HF_CHECKPOINT=c['policy']['hf_checkpoint'],REF_LOAD=c['policy']['distributed_checkpoint'],
        PROMPT_DATA=run/'train.jsonl',SAVE_ROOT=run/'checkpoints',LOG_DIR=run/'logs',RUN_NAME=run.name,
        HAPT_TRAIN_ENTRY=root/'src/harness_s/train_entry.py',
        HARNESS_S_ROOT=root,HARNESS_S_RUN=run,HAPT_PROJECT_ROOT=root,HAPT_JOINT_RUN_ROOT=run,
        HAPT_GRPO_ALGORITHM_VERSION='legacy-grpo-v1',HAPT_CASE_FAILURE_POLICY='zero',
        HAPT_DOCKER_BIN='/usr/bin/docker',HAPT_NODE_ROOT=r['node_root'],HAPT_NODE_RUNTIME_ROOT=r['node_root'],
        HARNESS_S_CONTROL_PYTHON='/opt/harness-s-python',HAPT_SANDBOX_NETWORK=r['docker_network'],
        ACTOR_NUM_NODES=1,ACTOR_NUM_GPUS_PER_NODE=len(r['gpu_ids']),RAY_NUM_GPUS_PER_NODE=len(r['gpu_ids']),
        TP_SIZE=r['actor_tp'],CP_SIZE=r['actor_cp'],MAX_TOKENS_PER_GPU=(s['context_window']+r['actor_cp']-1)//r['actor_cp'],
        CONTEXT_WINDOW=s['context_window'],ROLLOUT_MAX_CONTEXT_LEN=s['context_window'],ROLLOUT_MAX_RESPONSE_LEN=s['max_tokens'],
        ROLLOUT_NUM_GPUS_PER_ENGINE=r['rollout_gpus_per_engine'],SGLANG_MEM_FRACTION_STATIC=r['sglang_mem_fraction_static'],
        SGLANG_SERVER_CONCURRENCY=r['sglang_server_concurrency'],ROLLOUT_TEMPERATURE=s['temperature'],
        TRAIN_SEED=c['train_seed'],ROLLOUT_SEED=c['seed'],NUM_ROLLOUT=c['schedule']['steps'],
        ROLLOUT_BATCH_SIZE=c['schedule']['batch_size'],N_SAMPLES_PER_PROMPT=8,GLOBAL_BATCH_SIZE=c['schedule']['batch_size']*8,
        NORMALIZE_ADVANTAGES=1,USE_TIS=1,USE_KL_LOSS=0,KL_COEF=0,KL_LOSS_COEF=0,ENTROPY_COEF=0,
        EPS_CLIP=o['eps_clip'],EPS_CLIP_HIGH=o['eps_clip_high'],LR=o['lr'],ADAM_BETA1=o['beta1'],ADAM_BETA2=o['beta2'],WEIGHT_DECAY=o['weight_decay'],
        SAVE_INTERVAL=999999,SAVE_OPTIMIZER=1,USE_WANDB=0,LOSS_MASK_TYPE='qwen3_5',TOKEN_BUILD_MODE='tito',
        TOKEN_BUILD_MODEL='qwen3_5',RECORD_TOKEN_VERSIONS=1,ROLLOUT_SHUFFLE=0,
        DRESSAGE_SANDBOX_PROVIDER='docker',DRESSAGE_PADDOCK_MODE='blackbox',DRESSAGE_DOCKER_POOL_SIZE=0,
        DRESSAGE_CUSTOM_GENERATE_FUNCTION_PATH='dressage.rollout.generate.blackbox_dispatch.generate',
        DRESSAGE_REWARD_MODULES='dressage.recipes.dsh_environment.shared_reward',
        DRESSAGE_BLACKBOX_MAX_STEPS=30,DRESSAGE_PROXY_MAX_STEPS_PER_SESSION=32,PROXY_MAX_OUTPUT_TOKENS=s['max_tokens'],
        DRESSAGE_ROLLOUT_TRAJECTORY_CONCURRENCY=r['concurrency'],DRESSAGE_REWARD_CONCURRENCY=r['concurrency'],
        DRESSAGE_REQUIRE_COMPLETE_GROUPS=1,DRESSAGE_SYNC_MAX_REPLACEMENT_GROUPS=0,
        DRESSAGE_ROLLOUT_MAX_RETRIES=2,DRESSAGE_RAY_STOP_ON_EXIT=0,
        RAY_PORT=26379,RAY_DASHBOARD_PORT=28265,RAY_TEMP_DIR='/tmp/harness-s-ray',
        PROXY_PORT=28000,SGLANG_ROUTER_PORT=28001,SLIME_ROLLOUT_PORT_BASE=29000)
    env.update({key:os.environ[key] for key in ('HARNESS_S_ACTION','HARNESS_S_CHECKPOINT','HARNESS_S_TEST_HARNESS') if key in os.environ})
    state = run/'checkpoints/state.json'
    step = 0
    if state.exists():
        saved = read_json(state)
        step = saved['latest']
        if os.environ.get('HARNESS_S_ACTION')=='test':
            point = os.environ.get('HARNESS_S_CHECKPOINT','best')
            step = saved['best']['step'] if point=='best' else saved['final']
            if step is None:
                raise ValueError('Final checkpoint is not committed yet')
    elif os.environ.get('HARNESS_S_ACTION')=='test':
        raise ValueError('No committed training checkpoint')
    if step>0:
        sampler = run/f'checkpoints/rollout/dressage_data_source_state_{step-1}.pkl'
        if not sampler.is_file():
            raise FileNotFoundError(f'Committed sampler state is missing: {sampler}')
        env.update(RESUME_FROM_CKPT=1,RESUME_CKPT_ROOT=run/'checkpoints',RESUME_CKPT_STEP=step-1,RESUME_LOAD_OPTIM=1)
    env['DRESSAGE_EXTRA_RUNTIME_ENV_KEYS'] = ','.join(list(env)+['DOCKER_HOST','PYTHONPATH']+api_key_envs(c))
    return env


if __name__=='__main__':
    for key,value in settings().items():
        print('export '+key+'='+shlex.quote(str(value)))
