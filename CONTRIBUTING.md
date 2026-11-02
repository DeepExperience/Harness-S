# Contributing

Keep all training variants on the shared loop. Put environment differences in adapters. Keep task verifiers and execution limits consistent across baseline and sibling harnesses. Do not include API keys, private endpoints, model weights, run outputs or benchmark solutions.

Run `pytest -q` and `python scripts/check_release.py`. GPU integration changes should also report the exact recipe, checkpoint, inference placement and a short actual restart run. A dry-run launch does not count as a completed training test.
