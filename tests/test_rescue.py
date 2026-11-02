import asyncio
from copy import deepcopy
from types import SimpleNamespace
import pytest
from harness_s.rescue import replace_all_failure


def samples(rewards):
    return [SimpleNamespace(reward=r,group_index=9,index=i,session_id='old',metadata={
        'sampling_clone_index':i,'harness_id':'h0','task_id':'task',
        'backend_options':{'runtime':{},'proxy':{'sampling_seed_base':100+i}}}) for i,r in enumerate(rewards)]


def bindings():
    return {h:dict(id=h,path='/'+h,sha256=h) for h in ['h0']+[f'h{i}' for i in range(1,9)]}


async def _async_test_partial_success_never_replaced(method):
    group=samples([1]+[0]*7)
    calls=[]
    async def generate(items):
        calls.append(items)
        return items
    async def choose():
        pytest.fail('Router must not run for a successful base group')
    result,audit=await replace_all_failure(group,generate,choose,bindings(),method)
    assert len(calls)==1 and not audit['triggered']
    assert [s.reward for s in result]==[1]+[0]*7
    assert all(s.metadata['harness_id']=='h0' for s in result)


async def _async_test_rescue_replaces_base_and_preserves_group_and_context():
    group=samples([0]*8)
    calls=[]
    async def generate(items):
        calls.append(deepcopy(items))
        if len(calls)==2:
            items[0].reward=1
        return items
    async def choose():
        return [f'h{i}' for i in range(1,9)]
    result,audit=await replace_all_failure(group,generate,choose,bindings(),'harness_s')
    assert len(calls)==2 and audit['triggered']
    assert len(result)==8 and len(set(s.metadata['harness_id'] for s in result))==8
    assert [s.group_index for s in result]==[9]*8
    assert [s.index for s in result]==list(range(8))
    assert result[0].metadata['backend_options']['runtime']['harness_overlay']=='/h1'
    assert result[0].metadata['backend_options']['proxy']['sampling_seed_base']==1000000100
    assert result[0].session_id is None
    assert group[0].metadata['backend_options']['proxy']['sampling_seed_base']==100


async def _async_test_baseline_retains_all_failures():
    group=samples([0]*8)
    async def generate(items):return items
    async def choose():pytest.fail('Baseline must not route')
    result,audit=await replace_all_failure(group,generate,choose,bindings(),'grpo')
    assert not audit['triggered'] and len(result)==8


async def _async_test_repeated_siblings_rejected():
    async def generate(items):return items
    async def choose():return ['h1']*8
    with pytest.raises(ValueError,match='distinct'):
        await replace_all_failure(samples([0]*8),generate,choose,bindings(),'harness_s')


@pytest.mark.parametrize('method',['grpo','h0_rescue','harness_s'])
def test_partial_success_never_replaced(method):
    return asyncio.run(_async_test_partial_success_never_replaced(method))

def test_rescue_replaces_base_and_preserves_group_and_context():
    return asyncio.run(_async_test_rescue_replaces_base_and_preserves_group_and_context())

def test_baseline_retains_all_failures():
    return asyncio.run(_async_test_baseline_retains_all_failures())

def test_repeated_siblings_rejected():
    return asyncio.run(_async_test_repeated_siblings_rejected())
