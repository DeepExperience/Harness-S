"""AppWorld row construction; reward is always the official action replay."""
import hashlib
import json
import sys
from pathlib import Path
from .prompt import task_prompt


def rows(root, config, partition, bindings, runtime):
    root = Path(root)
    task_root = Path(config['appworld']['root']).resolve()
    task_python = Path(config['appworld']['python']).resolve()
    model = config['policy']['hf_checkpoint']
    output = []
    for task_id in (root / f'data/appworld/{partition}.txt').read_text().splitlines():
        spec = json.loads((task_root / 'data/tasks' / task_id / 'specs.json').read_text())
        task = dict(task_id=task_id, instruction=spec['instruction'], supervisor=spec['supervisor'])
        seed = int.from_bytes(hashlib.sha256(f"appworld|{config['seed']}|{task_id}".encode()).digest()[:4], 'big')
        output.append(dict(prompt=task_prompt(task), label='',
            generate_function_path='dressage.rollout.generate.blackbox_dispatch.generate',
            metadata=dict(instance_id='appworld-'+task_id, task_id=task_id,
                task_instruction=spec['instruction'], task_source='appworld', task_split=partition,
                experiment_partition=partition, agent_mode='blackbox', blackbox_type='deepseek_harness',
                harness_id='h0', harness_version='h0', harness_sha256=bindings['h0']['sha256'],
                reward_fn='appworld_shared_grpo', sampling_seed_mode='task-clone-v1',
                sandbox_image=config['runtime']['sandbox_image'], sandbox_timeout_sec=1800,
                sandbox_readonly_mounts=[str(root), str(task_root), str(task_python.parents[1]),
                    str(Path(config['runtime']['node_root']).resolve())],
                backend_options=dict(harness_id='task-environment-native', working_directory='/tmp',
                    model=dict(id=Path(model).name, context_window=config['sampling']['context_window'],
                        max_tokens=config['sampling']['max_tokens']), thinking='enabled',
                    permission_mode='danger-full-access',
                    runtime=dict(python_executable='/opt/harness-s-python',
                        dsh_executable=str(root/'runtime/dsh/node_modules/.bin/dsh'),
                        jsonrpc_executable=str(root/'runtime/unused-jsonrpc'),
                        minimal_composition=str(runtime/'minimal-piai.cordis.yml'),
                        model_overlay=str(runtime/'model-proxy-piai.cordis.yml'),
                        core_patch=str(runtime/'standard-core-native.cordis.yml'),
                        environment_patch=str(runtime/'task-environment-native.cordis.yml'),
                        harness_overlay=bindings['h0']['path']),
                    task_environment=dict(kind='appworld', python_executable=str(task_python),
                        server_script=str(root/'runtime/appworld/mcp_server.py'), data_root=str(task_root),
                        task_id=task_id, max_steps=30, reward_mode='binary'),
                    proxy=dict(max_steps=32, sampling_mode='force', sampling_temperature=(
                        config['sampling']['temperature'] if partition=='train' else 0),
                        sampling_top_p=1., sampling_seed_base=seed)))))
    return output
