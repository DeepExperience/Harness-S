# AppWorld sibling harness pool

This archive contains the complete frozen pool: **H0 and 19 auxiliary harnesses**. All share core fingerprint `b09d435aafa8cc2282bdaaa82224a9c69781bd1584fc800eebdb81569882d7b1`.

- `catalog.json`: stable member IDs, original version IDs, source content identities and public checksums.
- `<id>/harness.yml`: DSH composition overlay. `{{HARNESS_S_ROOT}}` is rendered during run preparation.
- `<id>/state.json`: original six-component configuration, preserved byte for byte.
- `<id>/skills/`: reusable skill content when present.
- `subsets/historical_top10.json`: exact subset from the historical Top-8 training run; separate from the full archive.

These are executable harness settings, not policy weights, task solutions or model behavior profiles. Profiles depend on the policy checkpoint and are regenerated from paired validation executions. The shared component runtime is `runtime/appworld/components.mjs`.

Verify all public artifacts with `harness-s check-pool`. Source identity hashes refer to the original archive; public artifact hashes cover the relocated release templates. YAML loader paths are the only location-dependent content changed during export.
