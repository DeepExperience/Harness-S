"""Launch an isolated training container without touching the host Ray cluster."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from .config import ROOT, api_key_envs
from .io import atomic_json, read_json, digest


def command(run, action="train", checkpoint="best", harness="h0"):
    run = Path(run).resolve()
    config = read_json(run/'config.json')
    for relative,expected in read_json(run/'source.json')['files'].items():
        if digest(ROOT/relative)!=expected:
            raise ValueError(f'Run source changed: {relative}; use a fresh run directory')
    gpu_ids = config['runtime']['gpu_ids']
    if not gpu_ids or len(set(gpu_ids))!=len(gpu_ids) or any(type(g) is not int or g<0 for g in gpu_ids):
        raise ValueError('Unique explicit GPU IDs required')
    docker = os.environ.get('DOCKER_BIN','docker')
    socket = os.environ.get('DOCKER_HOST','unix:///var/run/docker.sock')
    if not socket.startswith('unix://'):
        raise ValueError('Set DOCKER_HOST to a local Unix Docker socket')
    socket = socket.removeprefix('unix://')
    cmd = [docker,'run','--rm','--name','harness-s-'+run.name,'--init',
        '--gpus','"device='+','.join(map(str,gpu_ids))+'"',
        '--ipc','private','--shm-size','64g','--network',config['runtime']['docker_network'],
        '-w',str(ROOT),'-v',f'{socket}:/var/run/docker.sock',
        '-e','DOCKER_HOST=unix:///var/run/docker.sock','-e','HARNESS_S_ROOT='+str(ROOT),
        '-e','HARNESS_S_RUN='+str(run),'-e','HAPT_PROJECT_ROOT='+str(ROOT)]
    mounts = {ROOT,run,Path(config['appworld']['root']).resolve(),
        Path(config['appworld']['python']).resolve().parents[1],
        Path(config['policy']['hf_checkpoint']).resolve(),
        Path(config['policy']['distributed_checkpoint']).resolve(),
        Path(config['runtime']['node_root']).resolve()}
    for path in sorted(mounts):
        cmd += ['-v',f'{path}:{path}'+('' if path==run else ':ro')]
    for key in api_key_envs(config):
        cmd += ['-e',key]
    cmd += ['-e','HARNESS_S_ACTION='+action,'-e','HARNESS_S_CHECKPOINT='+checkpoint,
            '-e','HARNESS_S_TEST_HARNESS='+harness]
    cmd += [config['runtime']['training_image'],'bash','scripts/container_train.sh']
    return cmd


def launch(run, dry_run=False, action="train", checkpoint="best", harness="h0"):
    run = Path(run).resolve()
    cmd = command(run,action,checkpoint,harness)
    if dry_run:
        print(shlex.join(cmd))
        return
    config = read_json(run/'config.json')
    docker = cmd[0]
    if action=='test' and harness=='rank1' and config['method']!='harness_s':
        raise ValueError('Rank-1 evaluation requires a calibrated Harness-S run')
    if not (ROOT/'third_party/slime/train.py').exists():
        raise FileNotFoundError('Run bash scripts/bootstrap.sh first')
    for path in (config['policy']['hf_checkpoint'],config['policy']['distributed_checkpoint'],
                 config['appworld']['python'],config['runtime']['node_root']):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    for key in api_key_envs(config):
        if not os.environ.get(key):
            raise ValueError(f'Set the configured API key environment variable: {key}')
    network = config['runtime']['docker_network']
    probe = subprocess.run([docker,'network','inspect',network],capture_output=True)
    if probe.returncode:
        subprocess.run([docker,'network','create',network],check=True)
    # Docker gives this run a private PID namespace, network address and Ray head.
    attempts = config['runtime'].get('restart_attempts',2)+1
    for attempt in range(attempts):
        with (run/'supervisor.log').open('a') as log:
            process = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode==0:
            return
        atomic_json(run/f'restarts/attempt-{attempt}.json',dict(returncode=process.returncode,
            recovery_checkpoint=read_json(run/'checkpoints/state.json') if (run/'checkpoints/state.json').exists() else None))
    raise RuntimeError(f'Training exited {process.returncode}; inspect {run}/supervisor.log')
