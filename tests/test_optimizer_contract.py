"""Check the real backend converter and rewards with synthetic trajectories."""
from types import SimpleNamespace as NS
import pytest
from dressage.rollout.convert_samples import convert_samples_to_train_data
from dressage.training.grpo_algorithm import math_receipt, validate_args
from dressage.training.reward_post_process import reward_post_process

def samples(rewards, lengths=None, split=False):
    class Status:
        COMPLETED='completed'
        TRUNCATED='truncated'
    values=[]
    for group, reward_list in enumerate(rewards):
        for clone, reward in enumerate(reward_list):
            rid=group*8+clone
            length=lengths[rid] if lengths else 10
            pieces=[length//2, length-length//2] if split else [length]
            for segment, count in enumerate(pieces):
                values.append(NS(index=len(values),group_index=group,rollout_id=rid,
                    reward=reward if segment==len(pieces)-1 else 0.,
                    metadata=dict(task_id=f'task-{group}',instance_id=f'task-{group}',
                        parent_traj_id=f'parent-{rid}',segment_index=segment,sampling_clone_index=clone),
                    remove_sample=False,tokens=[0]*(count+1),response_length=count,loss_mask=[1]*count,
                    status=Status.COMPLETED,Status=Status,rollout_log_probs=None,train_metadata=None,
                    rollout_routed_experts=None,multimodal_train_inputs=None,teacher_log_probs=None))
    return values


def args(groups=2):
    return NS(advantage_estimator='grpo',rewards_normalization=True,grpo_std_normalization=True,
        n_samples_per_prompt=8,rollout_batch_size=groups,global_batch_size=groups*8,
        normalize_advantages=True,harness_mix='active_only',reward_key=None,
        calculate_per_token_loss=False,use_tis=True,use_rollout_logprobs=False,
        tis_clip_low=0.,tis_clip=2.,eps_clip=.2,eps_clip_high=.28,entropy_coef=0.,
        kl_coef=0.,use_kl_loss=False,num_steps_per_rollout=1)


@pytest.fixture(autouse=True)
def legacy(monkeypatch):
    monkeypatch.setenv('HAPT_GRPO_ALGORITHM_VERSION','legacy-grpo-v1')
    monkeypatch.delenv('DRESSAGE_CROSS_HARNESS_MODE',raising=False)


def test_actual_group_rewards_use_population_std():
    _,advantages=reward_post_process(args(),samples([[0]*8,[0]*7+[1]]))
    assert advantages[:8]==[0.]*8
    assert advantages[8:15]==pytest.approx([-1/7**.5]*7)
    assert advantages[15]==pytest.approx(7**.5)


@pytest.mark.parametrize('split',[False,True])
def test_task_token_denominators_survive_multi_segment_rollouts(split):
    aa=args(groups=1)
    ss=samples([[0]*7+[1]],[10,100]*4,split)
    data=convert_samples_to_train_data(aa,ss)
    receipt=math_receipt(aa,ss,data)
    rows=receipt['trajectories']
    assert rows[1]['nominal_weight']==pytest.approx(rows[0]['nominal_weight']*10)
    assert sum(x['nominal_weight'] for x in rows)==pytest.approx(1.)
    assert receipt['normalize_advantages'] is True
    assert receipt['use_tis'] is True


@pytest.mark.parametrize('key,value',[('use_tis',False),('normalize_advantages',False),
    ('kl_coef',.01),('use_kl_loss',True),('eps_clip_high',.2)])
def test_backend_rejects_recipe_drift(key,value):
    aa=args();setattr(aa,key,value)
    with pytest.raises(ValueError):validate_args(aa)
