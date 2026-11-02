"""Official action replay with the same fault policy in both GRPO arms."""
import sys
from pathlib import Path

from dressage.reward import register_reward
from . import reward as original

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts"))
from appworld_grpo_scoring import policy_argument_error, policy_failure, replay_with_retries


def replay_reward(sample, kind):
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("GRPO sample metadata missing")

    def checked_replay():
        value, diagnostics = original._replay(sample, expected_kind=kind)
        if type(diagnostics.get("success")) is not bool or value != float(diagnostics["success"]):
            raise ValueError("binary reward disagrees with official replay")
        return value, diagnostics

    try:
        (value, diagnostics), attempts, errors = replay_with_retries(checked_replay)
        diagnostics.update(replay_attempts=attempts, replay_retry_errors=errors)
    except Exception as exc:
        value = 0.
        if kind == "appworld" and policy_argument_error(exc):
            diagnostics = policy_failure(exc, len(original.extract_environment_actions(metadata.get("messages"))))
            diagnostics["replay_attempts"] = 1
        else:
            # Let the shared generation/scoring boundary attest, isolate and
            # record the case failure. A verifier outage is not an official zero.
            raise
    diagnostics["reward"] = value
    metadata[f"{kind}_blackbox_reward"] = diagnostics
    metadata[f"{kind}_dsh_reward"] = diagnostics
    return value


@register_reward("appworld_shared_grpo")
def appworld_reward(sample, *, args=None, **kwargs):
    return replay_reward(sample, "appworld")


@register_reward("alfworld_shared_grpo")
def alfworld_reward(sample, *, args=None, **kwargs):
    return replay_reward(sample, "alfworld")


# Content-identical legacy datasets can resume without changing their sampler
# fingerprint. The old name has no distinct scoring implementation.
register_reward("appworld_grpo_fixed_h0")(appworld_reward)
