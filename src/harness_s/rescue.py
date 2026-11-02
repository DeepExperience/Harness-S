"""Group replacement shared by every environment and training variant."""
from copy import deepcopy


def flatten(items):
    return [s for item in items for s in (item if isinstance(item, list) else [item])]


def terminal_outcomes(samples):
    groups = {}
    for sample in flatten(samples):
        meta = sample.metadata or {}
        clone = meta['sampling_clone_index']
        if clone not in groups or meta.get('segment_index',0) > (groups[clone].metadata or {}).get('segment_index',0):
            groups[clone] = sample
    result = []
    for clone, sample in sorted(groups.items()):
        reward = sample.reward
        if isinstance(reward, dict):
            reward = reward['reward']
        if reward not in (0,1):
            raise ValueError('Official binary reward required')
        result.append(float(reward))
    return result


def bind(sample, binding, fresh=False):
    sample = deepcopy(sample)
    sample.metadata.update(harness_id=binding['id'], harness_version=binding['id'], harness_sha256=binding['sha256'])
    sample.metadata['backend_options']['runtime']['harness_overlay'] = binding['path']
    if fresh:
        sample.session_id = None
        for key in ('session_id','parent_traj_id','last_failed_session_id'):
            sample.metadata.pop(key,None)
        # A separate random stream for the fresh episode; clone and group IDs remain fixed.
        sample.metadata['backend_options']['proxy']['sampling_seed_base'] += 1000000000
    return sample


async def replace_all_failure(group, generate, choose, bindings, method):
    initial = await generate([bind(s, bindings['h0']) for s in group])
    rewards = terminal_outcomes(initial)
    if len(rewards) != len(group):
        raise RuntimeError('Incomplete base group')
    if method == 'grpo' or any(rewards):
        return initial, dict(triggered=False, base=rewards)
    selected = ['h0']*len(group) if method=='h0_rescue' else await choose()
    if len(selected)!=len(group) or (method=='harness_s' and
            (len(set(selected))!=len(selected) or 'h0' in selected)):
        raise ValueError('Sibling rescue requires eight distinct auxiliary harnesses')
    rescued = await generate([bind(s,bindings[hid],fresh=True) for s,hid in zip(group,selected)])
    after = terminal_outcomes(rescued)
    if len(after)!=len(group):
        raise RuntimeError('Incomplete rescue group')
    return rescued, dict(triggered=True, base=rewards, rescue=after, selected=selected)
