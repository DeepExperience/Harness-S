"""One command contract for CPU admission, cached receipts and rollouts."""
from dressage.paddock.blackbox.execute_hooks import parse_blackbox_execute_cmds


def validate_command_schedule(metadata):
    try:
        schedule = parse_blackbox_execute_cmds(metadata.get('blackbox_execute_cmds'))
        if not schedule['before_agent']:
            raise ValueError('missing repository preparation commands')
        grades = [h for h in schedule['after_agent'] if h.name == 'swegym_harness']
        if len(grades) != 1:
            raise ValueError('exactly one swegym_harness grading command is required')
        for hooks in schedule.values():
            names = [h.name for h in hooks]
            if len(names) != len(set(names)):
                raise ValueError('duplicate command names within a stage')
        for hook in schedule['before_agent']:
            if hook.name in {'swegym_wsi_test_asset', 'swegym_legacy_build_runtime'} and not hook.required:
                raise ValueError(f'{hook.name} must be required')
        return schedule
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SWE-Gym command contract {metadata.get('instance_id', '?')}: {exc}") from exc
