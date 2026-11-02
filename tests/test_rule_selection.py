"""Exercise the public selection path without model credentials or downloads."""
import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from harness_s import backend, router
from harness_s.config import ROOT, api_key_envs, load
from harness_s.io import atomic_json, read_json
from harness_s.launch import command
from harness_s.rescue import replace_all_failure


@pytest.fixture
def recipe_env(monkeypatch):
    for key in ('POLICY_MODEL','POLICY_DIST_CHECKPOINT','APPWORLD_ROOT','APPWORLD_PYTHON','NODE_ROOT'):
        monkeypatch.setenv(key,'/tmp/harness-s-test/'+key)
    for key in ('ROUTER_API_KEY','ROUTER_BASE_URL','ROUTER_MODEL'):
        monkeypatch.delenv(key,raising=False)


@pytest.fixture
def no_models(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('The default selector must not instantiate a model client or LLM router')
    monkeypatch.setattr(router,'ModelClient',forbidden)
    monkeypatch.setattr(router,'Router',forbidden)
    monkeypatch.setitem(sys.modules,'sentence_transformers',None)


@pytest.mark.parametrize('size',['4b','9b'])
def test_default_recipe_and_launch_without_router_environment(size,recipe_env,tmp_path):
    config=load(ROOT/f'configs/appworld/harness_s_{size}.json')
    assert config['router']['mode']=='rules'
    assert config['profile_calibration']['enabled'] is False
    assert api_key_envs(config)==[]
    atomic_json(tmp_path/'config.json',config)
    atomic_json(tmp_path/'source.json',{'files':{}})
    rendered=command(tmp_path)
    assert 'ROUTER_API_KEY' not in rendered
    assert 'scripts/container_train.sh' in rendered


def test_omitted_switches_default_to_rules(recipe_env,tmp_path):
    config=read_json(ROOT/'configs/appworld/harness_s_4b.json')
    del config['router'],config['profile_calibration']
    atomic_json(tmp_path/'recipe.json',config)
    loaded=load(tmp_path/'recipe.json')
    assert loaded['router']=={'mode':'rules','neighbors':5}
    assert loaded['profile_calibration']=={'enabled':False}


def stage():
    return dict(step=10,success_rates={'h0':1.,'h1':.4,'h2':.9,'h3':.8},validation=[
        dict(task_id='shopping',instruction='buy amazon groceries',outcomes={'h0':1,'h1':1,'h2':0,'h3':1}),
        dict(task_id='music',instruction='play spotify music',outcomes={'h0':1,'h1':0,'h2':1,'h3':0})])


def test_nearby_success_precedes_global_success_and_breaks_ties(tmp_path,no_models):
    selector=router.make_selector({'neighbors':1},stage(),tmp_path)
    ranked=asyncio.run(selector.rank({'task_id':'query','instruction':'BUY amazon groceries'},['h2','h1','h3']))
    assert ranked==['h3','h1','h2']
    evidence=read_json(tmp_path/'query/ranking.json')
    assert evidence['similar_validation_tasks'][0]['task_id']=='shopping'
    assert evidence['local_success_rates']=={'h1':1,'h2':0,'h3':1}
    assert not evidence['fallback']
    inference=asyncio.run(selector.rank({'task_id':'test','instruction':'buy amazon groceries'},['h0','h1','h2','h3'],True))
    assert inference[0]=='h0'


@pytest.mark.parametrize('instruction',['unrelated vocabulary',''])
def test_no_similarity_uses_global_success(instruction,tmp_path,no_models):
    selector=router.make_selector({},stage(),tmp_path)
    ranked=asyncio.run(selector.rank({'task_id':'query','instruction':instruction},['h1','h2','h3']))
    assert ranked==['h2','h3','h1']
    assert read_json(tmp_path/'query/ranking.json')['fallback']


def test_shared_neighborhood_mean_and_deterministic_ties(tmp_path,no_models):
    evidence=stage()
    evidence['validation'][1]['instruction']='buy amazon groceries'
    evidence['success_rates']['h1']=evidence['success_rates']['h2']=.8
    selector=router.make_selector({'neighbors':5},evidence,tmp_path)
    ranked=asyncio.run(selector.rank({'task_id':'query','instruction':'buy amazon groceries'},['h3','h2','h1']))
    assert ranked==['h1','h2','h3']
    audit=read_json(tmp_path/'query/ranking.json')
    assert len(audit['similar_validation_tasks'])==2
    assert set(audit['local_success_rates'].values())=={.5}


def test_rule_top8_rescue(tmp_path,no_models):
    ids=['h0']+[f'h{i:02}' for i in range(1,20)]
    bindings={h:dict(id=h,path='/'+h,sha256=h) for h in ids}
    evidence=dict(step=0,success_rates={h:i/20 for i,h in enumerate(ids)},validation=[
        dict(task_id='v',instruction='buy groceries',outcomes={h:int(i<10) for i,h in enumerate(ids)})])
    selector=router.make_selector({},evidence,tmp_path)
    group=[SimpleNamespace(reward=0,index=i,group_index=1,session_id='old',metadata=dict(
        sampling_clone_index=i,task_id='query',backend_options=dict(runtime={},proxy=dict(sampling_seed_base=i))))
        for i in range(8)]
    calls=[]
    async def generate(items):
        calls.append(items)
        return items
    async def choose():
        return (await selector.rank(dict(task_id='query',instruction='buy groceries'),ids[1:]))[:8]
    result,audit=asyncio.run(replace_all_failure(group,generate,choose,bindings,'harness_s'))
    assert audit['selected']==[f'h{i:02}' for i in range(9,1,-1)]
    assert len(result)==8 and len(calls)==2
    assert all(x.session_id is None for x in result)


def install_eval_stubs(monkeypatch):
    for name,attributes in (
        ('slime.rollout.base_types',dict(RolloutFnEvalOutput=SimpleNamespace)),
        ('slime.utils.async_utils',dict(run=asyncio.run)),
    ):
        module=ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules,name,module)


def test_validation_collects_rates_without_profiles(tmp_path,monkeypatch,no_models):
    install_eval_stubs(monkeypatch)
    monkeypatch.delenv('HARNESS_S_ACTION',raising=False)
    config=dict(method='harness_s',profile_calibration=dict(enabled=False))
    bindings={'h0':{},'h1':{}}
    monkeypatch.setattr(backend,'context',lambda:(tmp_path,config,bindings))
    async def collect(args,config,root,active,partition):
        assert active==bindings and partition=='validation'
        return [dict(task_id='v',instruction='shopping',harness_id=h,reward=i,messages=[],infrastructure_failure=False)
                for i,h in enumerate(bindings)]
    monkeypatch.setattr(backend,'collect',collect)
    result=backend.evaluate(None,0,None)
    assert result.metrics=={'validation/h0':0.,'validation/h1':1.}
    saved=read_json(tmp_path/'router-stage.json')
    assert saved['profiles']=={}
    assert saved['validation'][0]['outcomes']=={'h0':0,'h1':1}


def test_rank1_evaluation_uses_rules(tmp_path,monkeypatch,no_models):
    install_eval_stubs(monkeypatch)
    monkeypatch.setenv('HARNESS_S_TEST_HARNESS','rank1')
    monkeypatch.setenv('HARNESS_S_CHECKPOINT','best')
    config=dict(method='harness_s',router=dict(mode='rules'),runtime=dict(concurrency=2),evaluation=dict(attempts=3))
    bindings={h:{} for h in stage()['success_rates']}
    monkeypatch.setattr(backend,'context',lambda:(tmp_path,config,bindings))
    atomic_json(tmp_path/'validation/step-0010/stage.json',stage())
    (tmp_path/'test.jsonl').write_text(json.dumps(dict(metadata=dict(task_id='test',task_instruction='buy amazon groceries')))+'\n')
    async def collect(args,config,root,bindings,partition,selected,attempts):
        assert selected=={'test':['h0']} and partition=='test' and attempts==3
        return [dict(task_id='test',reward=1)]*3
    monkeypatch.setattr(backend,'collect',collect)
    assert backend.test(None,10).metrics['pass_at_3']==1


def test_optional_backends_forward_key_names_only(recipe_env,tmp_path):
    config=load(ROOT/'configs/appworld/harness_s_4b.json')
    config['router']=dict(mode='llm',api_key_env='PROFILE_KEY',backends=[dict(api_key_env='VOTER_A'),dict(api_key_env='VOTER_B')])
    assert api_key_envs(config)==['VOTER_A','VOTER_B']
    config['profile_calibration']['enabled']=True
    assert api_key_envs(config)==['PROFILE_KEY','VOTER_A','VOTER_B']
    atomic_json(tmp_path/'config.json',config)
    atomic_json(tmp_path/'source.json',{'files':{}})
    rendered=command(tmp_path)
    for key in api_key_envs(config):
        assert rendered[rendered.index(key)-1]=='-e'


def test_profile_generation_remains_explicitly_available(tmp_path,monkeypatch):
    install_eval_stubs(monkeypatch)
    monkeypatch.delenv('HARNESS_S_ACTION',raising=False)
    config=dict(method='harness_s',profile_calibration=dict(enabled=True),router={})
    bindings={'h0':{},'h1':{}}
    monkeypatch.setattr(backend,'context',lambda:(tmp_path,config,bindings))
    async def collect(*args):
        return [dict(task_id='v',instruction='shopping',harness_id=h,reward=i,messages=[],infrastructure_failure=False)
                for i,h in enumerate(bindings)]
    async def profiles(paired,old,client):
        assert list(paired)==['h1'] and not old
        return {'h1':{'prefer':[{'when':'shopping','behavior':'check inventory'}],'avoid':[]}}
    monkeypatch.setattr(backend,'collect',collect)
    monkeypatch.setattr(router,'build_profiles',profiles)
    backend.evaluate(None,0,None)
    assert read_json(tmp_path/'router-stage.json')['profiles']['h1']['prefer']


def test_llm_voting_works_without_profiles(tmp_path,monkeypatch):
    import numpy as np
    evidence=stage()
    evidence['profiles']={}
    selector=router.make_selector({'mode':'llm','voters':6},evidence,tmp_path)
    selector.encoder=SimpleNamespace(encode=lambda *args,**kwargs:np.array([[1.,0.]]))
    selector.validation_embeddings=np.array([[1.,0.],[0.,1.]])
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=None))
    calls=[]
    async def call(name,instruction,data):
        calls.append(name)
        assert all(profile=={'prefer':[],'avoid':[]} for profile in data['profiles'].values())
        return {'ranked':['h3','h2','h1','h0']}
    monkeypatch.setattr(selector.client,'call',call)
    ranked=asyncio.run(selector.rank(dict(task_id='q',instruction='shopping'),['h0','h1','h2','h3']))
    assert ranked==['h3','h2','h1','h0'] and len(calls)==6
