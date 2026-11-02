# AppWorld release status

Verified locally on 2026-11-01:

- 19 CPU tests passed, including checks against the actual shared GRPO reward/conversion backend.
- All 20 original state files are byte-identical to the source archive. Public overlay templates change runtime locations only.
- All 732 frozen task IDs were prepared successfully against an installed AppWorld dataset.
- All 20 overlays render to run-local absolute paths. Training Docker command rendering passed without allocating GPUs.
- The included Slime patch applies cleanly to its pinned upstream commit.
- Python syntax, shell syntax and public source/private-path checks passed.

Not yet verified: building this portable image, real task-container execution through the new paths, GPU rollout/training, checkpoint restart and final evaluation. No new accuracy or throughput measurements are claimed. The full pool configuration differs from the recorded historical 10-member training subset.

The Git remote is DeepExperience/Harness-S. This local preparation has not published repository contents.
