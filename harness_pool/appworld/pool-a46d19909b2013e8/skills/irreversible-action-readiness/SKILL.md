---
name: irreversible-action-readiness
description: "Prepare irreversible writes by verifying target boundaries, mutable cart state, and required arguments."
whenToUse: "Use before a bulk mutation, cart order, or write with unresolved required IDs or quantities; do not use for already-resolved reads."
disable-model-invocation: false
user-invocable: false
---

1. Restate the requested outcome as an exact inclusion or exclusion predicate and expected target count. Use only task wording and authoritative reads, not assumptions.
2. For a bulk mutation, enumerate the complete candidate set from an authoritative source, apply the predicate, and verify the resulting target list and count before writing.
3. For a cart order, inspect the current cart in a separate read before ordering. Remove unrelated items with documented cart APIs, verify that the final cart contains only the requested items and quantities, and only then place the order.
4. For any write, inspect the target API schema once and map every required argument to an authoritative source. Do not fabricate IDs, card data, or credentials.
5. If a required dependency is absent after bounded authoritative checks, stop and report the missing data; do not retry unchanged or invent a value.
6. After the write, verify the requested state with an ordinary documented read when needed.
7. Do not use this skill for an already-resolved single-entity read, an ordinary information task, or a mutation whose target and arguments are already verified.
