"""Commit a complete restart point before pruning older checkpoints."""
import shutil
import logging
from pathlib import Path
from .io import atomic_json, read_json


def commit(directory, step, score, final=False):
    directory = Path(directory)
    path = directory/'state.json'
    state = read_json(path) if path.exists() else {'best':{'step':0,'score':-1},'latest':0}
    if score > state['best']['score']:
        state['best'] = dict(step=step,score=score)
    state['latest'] = step
    state['final'] = step if final else None
    atomic_json(path,state)
    keep = {state['best']['step'],step}
    # Only this run's completed checkpoints are candidates for deletion.
    for checkpoint in directory.glob('iter_*'):
        if checkpoint.is_dir() and checkpoint.name[5:].isdigit():
            completed_step = int(checkpoint.name[5:])+1
            if completed_step not in keep and completed_step<=step:
                try:
                    shutil.rmtree(checkpoint)
                except OSError as exc:
                    logging.warning('Checkpoint pruning will be retried: %s',exc)
    for sampler in (directory/'rollout').glob('dressage_data_source_state_*.pkl'):
        raw = sampler.stem.removeprefix('dressage_data_source_state_')
        if raw.isdigit() and int(raw)+1 not in keep and int(raw)+1<=step:
            try:
                sampler.unlink()
            except OSError as exc:
                logging.warning('Sampler pruning will be retried: %s',exc)
    return state
