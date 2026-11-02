"""Materialize a portable recipe before allocating GPUs."""
import json
import shutil
from pathlib import Path
from .config import ROOT, load
from .io import atomic_json, digest, read_json
from .pool import HarnessPool
from .environments.appworld import rows


def prepare(config_path, run_dir):
    config = load(config_path)
    run = Path(run_dir).resolve()
    if (run/'config.json').exists():
        if read_json(run/'config.json') != config:
            raise ValueError('Run already has a different frozen recipe; choose a new run directory')
        return run
    pool = HarnessPool(ROOT)
    pool.verify()
    split = read_json(ROOT/'data/appworld/manifest.json')
    ids = []
    for partition, count in {'train':320, 'validation':150, 'test':262}.items():
        path = ROOT/f'data/appworld/{partition}.txt'
        part = path.read_text().splitlines()
        if len(part) != count or digest(path) != split['sha256'][partition]:
            raise ValueError(f'Frozen split changed: {partition}')
        ids.extend(part)
    if len(set(ids)) != 732:
        raise ValueError('Train/Validation/Test task IDs must be disjoint')
    run.mkdir(parents=True, exist_ok=True)
    bindings = pool.render(run/'harnesses')
    runtime = run/'runtime'
    runtime.mkdir(exist_ok=True)
    for source in (ROOT/'runtime/appworld').glob('*.yml'):
        (runtime/source.name).write_text(source.read_text().replace('{{HARNESS_S_ROOT}}', str(ROOT)))
    for part in ('train','validation','test'):
        data = rows(ROOT, config, part, bindings, runtime)
        (run/f'{part}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in data))
    atomic_json(run/'bindings.json', bindings)
    atomic_json(run/'source.json', {'pool':digest(ROOT/'harness_pool/appworld/catalog.json'),
        'split':digest(ROOT/'data/appworld/manifest.json'),
        'files':{str(p.relative_to(ROOT)):digest(p) for base in ('src','runtime/appworld')
                 for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.mjs','.yml')}})
    atomic_json(run/'config.json', config)
    return run
