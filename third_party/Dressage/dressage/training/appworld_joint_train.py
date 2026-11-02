"""Compatibility entrypoint. All actor/rollout logic lives in policy_train."""
from dressage.training.policy_train import (
    train, parse_args, _binding, _controller, _save_validation_checkpoint,
)

if __name__ == "__main__":
    train(parse_args())
