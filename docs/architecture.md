# Architecture

`prepare.py` freezes the configuration and constructs AppWorld rows. `environments/appworld.py` owns task loading, prompts, environment options and the official replay contract. It does not own a training loop.

`backend.py` connects the shared orchestration to Dressage/Slime. `rescue.py` retains or replaces complete logical groups while preserving clone IDs, task IDs, generating contexts and token masks. The GRPO, H0-rescue and Harness-S methods select behavior through configuration.

`train.py` owns resident actor synchronization, validation boundaries and checkpoint commits. Actor parameters, Adam state, RNG and sampler state are saved before a restart point is committed. Resume restores the committed checkpoint and its calibration; partial later work is repeated from that point. `checkpoints.py` deletes old model and sampler files only after the new state is committed.

`router.py:RuleSelector` is the default selector. It retrieves five validation instructions with positive local TF-IDF cosine similarity, ranks candidates by mean success over this shared neighborhood, and breaks ties by overall validation success then harness ID. No positive matches fall back to overall validation success. It does not instantiate an API client or load an embedding model. Training selects eight distinct auxiliaries excluding H0; rank-1 evaluation includes H0. Validation statistics refresh at steps 0, 10, 20, 30 and 40 even when profiling is disabled.

With `profile_calibration.enabled=true`, `router.py` makes separate model calls for per-task differences, grouping and final prefer/avoid profiles. Distinct singleton cases remain eligible. Raw HTTP response bodies and processed outputs are saved alongside each calibration. API failures retain the previous profile, or an empty profile on the initial calibration. If every ranking ballot fails, validation success ordering is an explicitly logged fallback. These degradation paths are operational choices, not additional paper results.

With `router.mode="llm"`, the optional router uses task instructions, current profiles, validation success rates and the five nearest validation instructions under Qwen3-Embedding-4B. Rescue excludes H0; inference includes H0. Six independently ordered ballots from the same configured model (or round-robin `backends`) each rank four candidates, weighted 1, 3/4, 1/2, 1/4. Validation rate then harness ID break ties. Profile calibration is independently disabled by default; voting can run with empty profiles. Test outcomes never enter calibration.

The public pool stores portable templates; preparation writes location-specific rendered copies inside the run directory. Prompt/state/skill content remains traceable to the original pool. The shared V5 component runtime preserves how context, memory, hooks and tool-view settings are applied.

Upstream runtime code stays under `third_party/` with its license and provenance. The public API and method logic stay under `src/harness_s/`; historical experiment launchers, frozen output trees, model weights and credentials are excluded.
