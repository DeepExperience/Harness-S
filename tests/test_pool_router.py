import asyncio
import json
import shutil
from pathlib import Path
import pytest
from harness_s.config import ROOT
from harness_s.pool import HarnessPool
from harness_s.router import aggregate, build_profiles
from harness_s.checkpoints import commit


def test_pool_complete_and_relocatable(tmp_path):
    pool=HarnessPool(ROOT)
    assert pool.verify()==20
    assert len({h['core_sha256'] for h in pool.members.values()})==1
    rendered=pool.render(tmp_path/'with spaces'/'harnesses')
    for binding in rendered.values():
        text=Path(binding['path']).read_text()
        assert '{{HARNESS_S_ROOT}}' not in text
        assert '/mnt/' not in text.replace(str(ROOT),'')


def test_split_disjoint_and_exact():
    partitions=[(ROOT/f'data/appworld/{p}.txt').read_text().splitlines() for p in ('train','validation','test')]
    assert list(map(len,partitions))==[320,150,262]
    assert len(set(sum(partitions,[])))==732


def test_ballot_weights_and_ties():
    candidates=['h0','h1','h2','h3','h4']
    ranked=aggregate([{'ranked':['h2','h1','h0','h4']},{'ranked':['h1','h2','h4','h0']}],candidates,{'h1':.8,'h2':.4})
    assert ranked[:2]==['h1','h2'] and ranked[-1]=='h3'
    with pytest.raises(ValueError):aggregate([{'ranked':['h1']*4}],candidates,{})


async def _async_test_profile_calls_separate_all_three_stages(tmp_path):
    class Client:
        directory=tmp_path
        calls=[]
        async def call(self,name,instruction,data):
            self.calls.append(name)
            if '/differences/' in name:return {'differences':[{'when':'pagination','behavior':'reads all pages'}]}
            if name.endswith('/groups'):return {'groups':[{'cases':['task1','task2']}]}
            return {'prefer':[{'when':'pagination','behavior':'reads all pages'}],'avoid':[]}
    client=Client()
    result=await build_profiles({'h1':[{'task_id':'task1'},{'task_id':'task2'}]}, {},client)
    assert client.calls==['h1/differences/task1','h1/differences/task2','h1/groups','h1/profile']
    assert len(result['h1']['prefer'])==1


def test_retention_commits_before_deleting(tmp_path):
    for step in (10,20,30):
        (tmp_path/f'iter_{step-1:07d}').mkdir()
    commit(tmp_path,10,.6)
    commit(tmp_path,20,.5)
    assert (tmp_path/'iter_0000009').exists()
    commit(tmp_path,30,.7,final=True)
    assert list(p.name for p in tmp_path.glob('iter_*'))==['iter_0000029']
    saved=json.loads((tmp_path/'state.json').read_text())
    assert saved['best']['step']==saved['final']==30


def test_profile_calls_separate_all_three_stages(tmp_path):
    return asyncio.run(_async_test_profile_calls_separate_all_three_stages(tmp_path))
