"""One configuration schema for baseline, H0 resampling and sibling rescue."""
import os
from pathlib import Path
from .io import read_json

ROOT = Path(__file__).resolve().parents[2]


def model_configs(config):
    """Only active API clients need credentials, including optional voting backends."""
    if config['method'] != 'harness_s':
        return []
    router = config.get('router',{})
    clients = []
    if config.get('profile_calibration',{}).get('enabled',False):
        clients.append(router)
    if router.get('mode','rules') == 'llm':
        clients.extend([{**router,**backend} for backend in router.get('backends',[])] or [router])
    return clients


def api_key_envs(config):
    return sorted({client['api_key_env'] for client in model_configs(config)})


def load(path):
    value = read_json(path)
    def expand(item):
        if isinstance(item, str):
            result = os.path.expandvars(item)
            if '${' in result:
                raise ValueError(f'Unset environment variable in {item}')
            return result
        if isinstance(item, list):
            return [expand(x) for x in item]
        if isinstance(item, dict):
            return {k: expand(v) for k, v in item.items()}
        return item
    value = expand(value)
    for section,key in [('policy','hf_checkpoint'),('policy','distributed_checkpoint'),('appworld','root'),('runtime','node_root')]:
        value[section][key] = str(Path(value[section][key]).expanduser().resolve())
    if value['environment'] != 'appworld':
        raise ValueError('This release includes the AppWorld adapter')
    if value['method'] not in ('grpo', 'h0_rescue', 'harness_s'):
        raise ValueError('Unknown method')
    if value['method'] == 'harness_s':
        router = value.setdefault('router',{})
        router.setdefault('mode','rules')
        router.setdefault('neighbors',5)
        profile = value.setdefault('profile_calibration',{})
        profile.setdefault('enabled',False)
        if type(profile['enabled']) is not bool:
            raise ValueError('profile_calibration.enabled must be a boolean')
        if router['mode'] not in ('rules','llm'):
            raise ValueError('router.mode must be rules or llm')
        if type(router['neighbors']) is not int or router['neighbors'] < 1:
            raise ValueError('router.neighbors must be a positive integer')
        for client in model_configs(value):
            for key in ('base_url','model','api_key_env'):
                if not client.get(key):
                    raise ValueError(f'Active LLM client requires router.{key}')
        if router['mode'] == 'llm':
            router.setdefault('voters',6)
            router.setdefault('embedding_model','Qwen/Qwen3-Embedding-4B')
            if type(router['voters']) is not int or router['voters'] < 1:
                raise ValueError('router.voters must be a positive integer')
    if value['sampling']['group_size'] != 8:
        raise ValueError('The paper recipe requires G=8')
    if value['kl'] != 0:
        raise ValueError('This recipe has no reference model or KL penalty')
    if value['schedule']['steps'] * value['schedule']['batch_size'] > 320:
        raise ValueError('Training schedule exceeds the frozen training partition')
    if value['he']['enabled']:
        raise ValueError('This release uses the frozen offline pool; online HE is not a training stage')
    if value['runtime']['rollout_gpus_per_engine'] < 1:
        raise ValueError('Invalid inference GPU allocation')
    return value
