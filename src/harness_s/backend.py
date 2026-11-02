"""Dressage/Slime hooks: shared sampling, replay, evaluation and rescue."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from .io import atomic_json, read_json, telemetry
from .rescue import bind, flatten, replace_all_failure, terminal_outcomes


def context():
    root = Path(os.environ['HARNESS_S_RUN'])
    return root, read_json(root/'config.json'), read_json(root/'bindings.json')


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        return evaluate(args,rollout_id,data_source,evaluation=True)
    from dressage.rollout import sync_rollout as sync
    from dressage.rollout.generate.runtime import generate_group
    from .router import make_selector
    root, config, bindings = context()
    stage = read_json(root/'router-stage.json') if config['method']=='harness_s' else None
    selector = make_selector(config.get('router',{}),stage,root/f'router/step-{rollout_id+1:04d}') if stage else None
    original = sync._submit_group
    async def submit(args,group,state,pendings,task_to_group):
        async def operation():
            async def wave(inputs):
                return await generate_group(sync.generate_and_rm_group,args,inputs,state.sampling_params.copy())
            async def choose():
                sample = group[0]
                ranked = await selector.rank(dict(task_id=sample.metadata['task_id'],
                    instruction=sample.metadata['task_instruction']),[h for h in bindings if h!='h0'])
                return ranked[:len(group)]
            result,audit = await replace_all_failure(group,wave,choose,bindings,config['method'])
            telemetry(root/f'rescue/step-{rollout_id+1:04d}/{group[0].metadata["task_id"]}.json',audit)
            return flatten(result)
        task = asyncio.create_task(operation())
        pendings.add(task)
        task_to_group[task] = group
    sync._submit_group = submit
    try:
        return sync.generate_rollout_sync(args,rollout_id,data_source)
    finally:
        sync._submit_group = original


async def collect(args, config, root, bindings, partition, selected=None, attempts=1):
    from slime.utils.types import Sample
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
    from dressage.rollout.generate.runtime import generate_group
    from dressage.rollout.sync_rollout import _resume_proxy_after_sync_boundary, _pause_proxy_for_natural_drain, _wait_for_rollout_servers_to_drain
    state = GenerateState(args)
    tasks = [json.loads(line) for line in (root/f'{partition}.jsonl').read_text().splitlines()]
    limit = asyncio.Semaphore(config['runtime']['concurrency'])
    async def one(index,row,hid,clone):
        async with limit:
            meta = deepcopy(row['metadata'])
            meta['sampling_clone_index'] = clone
            meta['backend_options']['proxy']['sampling_temperature'] = 0 if partition=='validation' else config['evaluation']['temperature']
            meta['backend_options']['proxy']['sampling_seed_base'] += clone
            sample = Sample(index=index,group_index=index,prompt=row['prompt'],label='',metadata=meta)
            sample.generate_function_path = row['generate_function_path']
            sample = bind(sample,bindings[hid])
            result = flatten(await generate_group(generate_and_rm_group,args,[sample],state.sampling_params.copy()))
            reward = terminal_outcomes(result)[0]
            anchor = max(result,key=lambda s:(s.metadata or {}).get('segment_index',0))
            return dict(task_id=meta['task_id'],instruction=meta['task_instruction'],harness_id=hid,
                clone=clone,reward=reward,messages=anchor.metadata.get('messages',[]),
                infrastructure_failure=bool(anchor.metadata.get('case_failure_diagnostics')))
    await _resume_proxy_after_sync_boundary()
    try:
        jobs = []
        for index,row in enumerate(tasks):
            choices = selected[row['metadata']['task_id']] if selected else list(bindings)
            jobs += [one(index*len(bindings)*attempts+j*attempts+k,row,hid,k)
                     for j,hid in enumerate(choices) for k in range(attempts)]
        return await asyncio.gather(*jobs)
    finally:
        await _pause_proxy_for_natural_drain()
        await _wait_for_rollout_servers_to_drain(args)
        state.reset()


def evaluate(args, rollout_id, data_source, evaluation=True):
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.utils.async_utils import run
    from .router import ModelClient, build_profiles
    root,config,bindings = context()
    if os.environ.get('HARNESS_S_ACTION') == 'test':
        return test(args,rollout_id)
    async def calibration():
        active = bindings if config['method']=='harness_s' else {'h0':bindings['h0']}
        observed = await collect(args,config,root,active,'validation')
        folder = root/f'validation/step-{rollout_id:04d}'
        atomic_json(folder/'trajectories.json',observed)
        by_hid = {h:{x['task_id']:x for x in observed if x['harness_id']==h} for h in active}
        rates = {h:sum(x['reward'] for x in cases.values())/len(cases) for h,cases in by_hid.items()}
        stage = dict(step=rollout_id,success_rates=rates,profiles={},
            validation=[dict(task_id=tid,instruction=x['instruction'],
                outcomes={h:cases[tid]['reward'] for h,cases in by_hid.items()}) for tid,x in by_hid['h0'].items()])
        if config['method']=='harness_s' and config.get('profile_calibration',{}).get('enabled',False):
            paired = {h:[dict(task_id=tid,instruction=x['instruction'],sibling=x,base=by_hid['h0'][tid],
                infrastructure_failure=x['infrastructure_failure'] or by_hid['h0'][tid]['infrastructure_failure'])
                for tid,x in cases.items()] for h,cases in by_hid.items() if h!='h0'}
            old = read_json(root/'router-stage.json')['profiles'] if (root/'router-stage.json').exists() else {}
            stage['profiles'] = await build_profiles(paired,old,ModelClient(config['router'],folder/'profile-api'))
        atomic_json(folder/'stage.json',stage)
        atomic_json(root/'router-stage.json',stage)
        return rates
    rates = run(calibration())
    return RolloutFnEvalOutput(data={},metrics={f'validation/{h}':value for h,value in rates.items()})


def test(args, step):
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.utils.async_utils import run
    from .router import make_selector
    root,config,bindings = context()
    harness = os.environ.get('HARNESS_S_TEST_HARNESS','h0')
    point = os.environ.get('HARNESS_S_CHECKPOINT','best')
    folder = root/f'test/{point}-{harness}'
    async def execute():
        tasks = [json.loads(line) for line in (root/'test.jsonl').read_text().splitlines()]
        selected = {x['metadata']['task_id']:['h0'] for x in tasks}
        if harness=='rank1':
            stage = read_json(root/f'validation/step-{step:04d}/stage.json')
            selector = make_selector(config.get('router',{}),stage,folder/'routing')
            limit = asyncio.Semaphore(config['runtime']['concurrency'])
            async def route(row):
                async with limit:
                    meta = row['metadata']
                    ranked = await selector.rank(dict(task_id=meta['task_id'],instruction=meta['task_instruction']),
                                               list(bindings),inference=True)
                    selected[meta['task_id']] = ranked[:1]
            await asyncio.gather(*(route(row) for row in tasks))
        atomic_json(folder/'selection.json',selected)
        observed = await collect(args,config,root,bindings,'test',selected,config['evaluation']['attempts'])
        atomic_json(folder/'trajectories.json',observed)
        by_task = {tid:[x['reward'] for x in observed if x['task_id']==tid] for tid in selected}
        metrics = dict(pass_at_3=sum(max(x) for x in by_task.values())/len(by_task),
                       avg_at_3=sum(sum(x)/len(x) for x in by_task.values())/len(by_task),
                       tasks=len(by_task),attempts=config['evaluation']['attempts'],checkpoint_step=step)
        atomic_json(folder/'metrics.json',metrics)
        return metrics
    return RolloutFnEvalOutput(data={},metrics=run(execute()))
