# AppWorld reproduction

## Required inputs

Install AppWorld using its upstream instructions in a separate environment. `APPWORLD_ROOT` must contain `data/tasks/<task_id>/specs.json` and the databases needed by the official `AppWorld` executor. This repository distributes IDs and family assignments only. Task database licensing remains upstream's.

Download Qwen/Qwen3.5-4B or Qwen/Qwen3.5-9B. `POLICY_MODEL` points to its local Hugging Face directory. `POLICY_DIST_CHECKPOINT` points to the matching converted Megatron checkpoint. Architecture flags are under `third_party/model_configs/`.

The pinned Slime converter is `third_party/slime/tools/convert_hf_to_torch_dist.py`. Run it in the GPU training image with the matching model flags, rather than using a checkpoint converted for a different architecture:

```bash
bash scripts/convert_checkpoint.sh qwen3.5-4B
# Or: bash scripts/convert_checkpoint.sh qwen3.5-9B
# CONVERSION_GPU selects the explicitly allocated conversion GPU (default 0).
```

`NODE_ROOT/bin/node` must be a real supported Node installation, readable inside task containers. Bootstrap installs the pinned DSH package under `runtime/dsh`. Default recipes need no router endpoint or API key. Raw API request logs, when the optional model path is enabled, omit authorization headers.

## Method mapping

| Manuscript component | Implementation |
| --- | --- |
| Sibling pool, fixed after offline evolution | `harness_pool/appworld/catalog.json`, `pool.py` |
| All-failure group replacement | `rescue.py:replace_all_failure` |
| Group-relative reward and token whitening | Adapted Dressage `training/reward_post_process.py` |
| Task-token denominator, generating context, policy-token mask | Adapted Dressage `rollout/convert_samples.py` and pinned Slime patch |
| PPO ratio clipping and detached TIS | Pinned Slime Megatron loss; common backend flags |
| Default rule-based Top-8 rescue | `router.py:RuleSelector` |
| Optional paired validation and three-stage profile refresh | `backend.py:evaluate`, `router.py:build_profiles` |
| Optional weighted rank voting | `router.py:aggregate` |
| H0 / rank-1 evaluation with three fixed-harness attempts | `backend.py:test` |

## Protocol boundaries

The included task order is the frozen 320/150/262 split. Task IDs are disjoint; families can overlap. The same 150 validation tasks select the checkpoint and provide per-task harness outcomes for rule-based selection (and optional profiles). Checkpoint selection uses H0 validation success in this public recipe. The archived historical run selected by a different fixed-pool metric; this distinction is intentional and is not a claim of exact score reproduction.

Training uses one seed, batch size 8, G=8 and temperature 1. Validation uses temperature 0 and one attempt per task and harness at steps 0, 10, 20, 30 and 40. Profile generation is disabled by default. Final policy evaluation uses three attempts at temperature 1, reporting both Pass@3 and Avg@3. The paper's averages over three independent runs need three explicitly launched runs; no extra runs are started automatically.

The historical Top-8 experiment archived here executed a 10-member subset. The default public recipe uses all 20, matching the manuscript's stated pool. The subset membership is recorded separately, with no test-based selection. Historical profiles are not imported as if they belonged to a newly trained checkpoint.

The public recipe uses one GPU per rollout engine and 32 concurrent tasks. Historical experiments sometimes used two GPUs per engine. If a 9B instance cannot fit the requested context, explicitly set `runtime.rollout_gpus_per_engine` to 2 and document the change. Actor tensor/context parallelism remains separate from inference instance placement.

## Optional LLM routing and profiles

The 4B and 9B public Harness-S recipes default to:

```json
"profile_calibration": {"enabled": false},
"router": {"mode": "rules", "neighbors": 5}
```

Rules tokenize case-folded instructions into words, fit TF-IDF on validation instructions only, and retrieve up to five positive cosine matches. All candidates use the same nearby tasks: mean success there is the primary ranking key, overall validation success breaks ties, and harness ID breaks remaining ties. If no instructions overlap, selection uses overall validation success. This inexpensive lexical similarity does not capture semantic paraphrases as an embedding model would. Neither profiles nor model calls participate in rule selection. Validation rollouts remain necessary and still incur policy execution cost.

To opt into the previous model-based strategy, replace those sections in a new recipe with:

```json
"profile_calibration": {"enabled": true},
"router": {
  "mode": "llm",
  "base_url": "${ROUTER_BASE_URL}",
  "model": "${ROUTER_MODEL}",
  "api_key_env": "ROUTER_API_KEY",
  "temperature": 0.0,
  "concurrency": 32,
  "voters": 6,
  "neighbors": 5,
  "embedding_model": "Qwen/Qwen3-Embedding-4B"
}
```

Set the three `ROUTER_*` environment variables before preparation/launch. Profiling and voting are independent switches; LLM voting also supports disabled profiles. Six voters mean six independently ordered calls to the same model by default, not six different models. An optional `router.backends` list overrides `base_url`, `model`, and/or `api_key_env` per voting backend in round-robin order. Profile generation uses the top-level router client. All active key environment variables are forwarded by name to Docker and Ray; key values are not embedded in launch commands or frozen JSON configuration. `.env` and `outputs/` are ignored, and `.env.example` contains an empty key placeholder.

Use a new run directory for a changed recipe or changed source files; existing runs freeze both. The rule-based default changes the historical routing protocol and has no newly measured accuracy claim.

## Recovery and retention

Rerun the same training command and run directory to resume. Configuration and source identities must match. The loop saves full actor/optimizer/RNG/sampler state at validation boundaries. Only committed checkpoints can be resumed. A failure between boundaries loses the incomplete interval; it does not produce a partial resumable model.

Checkpoint metadata, metrics and trajectories remain after pruning. Model and sampler payloads are limited to the validation-best and latest committed restart points, then best and final. The cleanup routine only removes checkpoints inside the current run.

Optional telemetry and profile calls can fail without stopping an optimizer run. Missing model/data inputs, invalid group contracts, numerical corruption and unavailable critical checkpoint state remain errors. Infrastructure failures are recorded separately from official task failures and excluded from behavioral profile evidence.

## Validation status

CPU tests cover archive checksums, disjoint task IDs, relocatable overlays, all-failure replacement, clone/group identity, fresh seeds, rule-based selection without API credentials, voting, three separate profiling stages and checkpoint retention. Preparation and command rendering are checked against the available AppWorld installation without running a GPU workload.

The reorganized portable launcher has **not completed a GPU end-to-end run**. Remaining release validation is a short GPU run covering actual Docker task startup, TITO token/mask alignment, an optimizer update, validation and optional profile calibration, checkpoint restart and final H0/routed evaluation. Passing CPU tests is not evidence that all of those integration boundaries are working.
