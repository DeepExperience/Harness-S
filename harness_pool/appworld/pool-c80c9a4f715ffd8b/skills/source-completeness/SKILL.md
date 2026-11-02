---
name: source-completeness
description: "Establish authoritative source boundaries, exact relationship membership, and observed values before acting on AppWorld list or derived-value results."
whenToUse: "Use when a task depends on all members, a last-N item, a relationship category, a bounded date window, an exact set intersection, or a value that must be derived from an authoritative source. Do not use after a unique target or definitive boundary is already resolved."
disable-model-invocation: false
user-invocable: false
---

Purpose: establish authoritative source completeness and exact membership before filtering, joining, counting, updating, deriving a value, or causing side effects.

Required inputs:
- The exact requested predicate, including relationship, category, date field, exception clauses, and whether boundaries are inclusive.
- Detailed API documentation for every endpoint that can supply the predicate or value.
- The observed current date for a relative window, obtained from the environment or a documented current-date API.
- The page parameters, sort order, filters, and result count for each relevant list call.

Procedure:
1. Name the target set or value before fetching. Identify each authoritative source and the participant or entity field to compare. Do not replace a relationship with a superficial status, a container field with a per-entity field, or a referenced request with an unverified collection.
2. Read detailed endpoint documentation before relying on a list result. Record default and maximum page size, page index semantics, sort order, supported filters, date field, and terminal condition. Do not treat a default or bounded page as exhaustive.
3. For a relative date window, observe the current date explicitly. Compute both inclusive boundaries from that observation, print them, and filter on the documented date field. Never infer today or a period from the newest returned record.
4. When exhaustive traversal is required, request the largest documented page size and batch related page reads in one execute block when practical. Continue only until an empty page, a short page at the maximum page size, a documented terminal condition, a definitive filter boundary, or the unique requested target is resolved.
5. Preserve decision-critical identifiers and fields needed for the exact predicate. Build sets or unions explicitly. Intersect every required relationship or membership set; do not guess an entity, identifier, category, or value that was not returned. Deduplicate only when the task asks for unique members.
6. Derive values from the authoritative per-entity fields required by the predicate. Compute once, print the inputs and result, and do not substitute a placeholder for a missing source value.
7. Before a side effect or completion, verify the selected set against the stated predicate and boundaries. Confirm that every required source was considered and that no extra member from an unrequested relationship, category, or date was included. If evidence is incomplete, fetch the missing page, entity, or source rather than proceeding on an assumption.
8. Stop as soon as the boundary is definitive or the unique target is resolved. Do not force additional discovery after a short page, a documented terminal condition, a definitive filter boundary, or a resolved target. Preserve successful targeted-query and direct-read procedures.

Completion check: report or act on only the exact selected set or derived value. If a required source or boundary cannot be established with available APIs, state that limitation in the task result rather than inventing completeness.
