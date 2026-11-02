# Third-party notices

- **Dressage**, Accio-Lab, Apache-2.0. The adapted runtime is vendored in `Dressage/`; the upstream base commit is recorded in `versions.json`. Local changes include blackbox DSH execution, AppWorld official replay, multi-segment policy-token handling, group-relative rewards and task-token reduction. This export additionally replaces project-local runtime paths and uses the public training hooks. The original license is retained.
- **Slime**, THUDM, Apache-2.0. `versions.json` pins the upstream commit. `patches/slime.patch` preserves the working compatibility changes for this backend. The bootstrap script checks the base revision before applying the patch. Slime's own license is downloaded with its source.
- **DeepSeek Harness**, deepseek-ai, MIT. The npm release `0.1.0-rc.7` is pinned in `runtime/dsh/package.json`. Its license remains in the installed dependency. The project-specific composition overlays and component plugin are included in this release.
- **AppWorld**, StonyBrookNLP. Install and obtain benchmark data from upstream under its applicable terms. This repository includes the experiment task IDs and integration bridge; no task databases, answers or environment installation are bundled.

The repository's Apache-2.0 license applies to project code. It does not replace dependency, model or dataset licenses.
