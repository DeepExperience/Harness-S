"""Resident actor loop: all variants share sampling, optimizer and recovery."""
import os
from pathlib import Path
from .io import read_json, telemetry, atomic_json
from .checkpoints import commit


def train(args):
    import ray
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from slime.utils.logging_utils import configure_logger, init_tracking, finish_tracking
    from dressage.training.grpo_algorithm import validate_args
    configure_logger()
    validate_args(args)
    if args.use_critic or args.use_kl_loss or args.kl_coef or args.kl_loss_coef:
        raise ValueError('Actor-only GRPO with no reference-model KL is required')
    run = Path(os.environ['HARNESS_S_RUN'])
    config = read_json(run/'config.json')
    checkpoints = run/'checkpoints'
    boundaries = set(config['schedule']['validation_steps'])|{config['schedule']['steps']}
    init_tracking(args)
    manager = None
    try:
        placement = create_placement_groups(args)
        manager,_ = create_rollout_manager(args,placement['rollout'])
        actor,critic = create_training_models(args,placement,manager)
        if critic is not None:
            raise ValueError('Unexpected critic')
        def sync():
            if args.offload_rollout:
                ray.get(manager.onload_weights.remote())
            actor.update_weights()
            if args.offload_rollout:
                ray.get(manager.onload_kv.remote())
        def evaluate(step):
            ray.get(manager.eval.remote(step))
            stage = read_json(run/f'validation/step-{step:04d}/stage.json')
            return stage['success_rates']['h0']
        sync()
        start = int(args.start_rollout_id)
        if os.environ.get('HARNESS_S_ACTION') != 'test' and not (run/f'validation/step-{start:04d}/stage.json').exists():
            evaluate(start)
        if os.environ.get('HARNESS_S_ACTION') == 'test':
            ray.get(manager.eval.remote(start))
            return
        atomic_json(run/'router-stage.json',read_json(run/f'validation/step-{start:04d}/stage.json'))
        if start==0:
            commit(checkpoints,0,read_json(run/'router-stage.json')['success_rates']['h0'])
        for rollout_id in range(start,config['schedule']['steps']):
            step = rollout_id+1
            data = ray.get(manager.generate.remote(rollout_id))
            if args.offload_rollout:
                ray.get(manager.offload.remote())
            ray.get(actor.async_train(rollout_id,data))
            if step in boundaries:
                actor.save_model(rollout_id,force_sync=True)
                ray.get(manager.save.remote(rollout_id))
            if not args.offload_train:
                actor.clear_memory()
            sync()
            if step in boundaries:
                score = evaluate(step)
                commit(checkpoints,step,score,final=step==config['schedule']['steps'])
            telemetry(run/f'metrics/step-{step:04d}.json',dict(completed_updates=step))
    finally:
        if manager is not None:
            try:
                ray.get(manager.dispose.remote())
            except Exception:
                pass
        finish_tracking(args)


if __name__=='__main__':
    from slime.utils.arguments import parse_args
    train(parse_args())
