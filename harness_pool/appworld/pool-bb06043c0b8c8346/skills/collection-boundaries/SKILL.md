---
name: collection-boundaries
description: "Establish exhaustive collection boundaries, exact relationship membership, and explicit relative-date windows before acting on AppWorld list results."
whenToUse: "Use when a task depends on all members, a last-N item, a relationship category, a bounded date window, or updating a collection. Do not use after a unique target or definitive boundary is already resolved."
disable-model-invocation: false
user-invocable: false
---

Purpose: make the boundary of every decision-critical collection explicit before filtering, joining, counting, updating, or causing side effects.

Required inputs:
- The exact requested membership predicate, including relationship, category, date field, and whether boundaries are inclusive.
- Detailed API documentation for each list endpoint that can supply the predicate.
- The observed current date for a relative window, obtained from the environment or a documented current-date API.
- The page parameters, sort order, filters, and result count for each relevant list call.

Procedure:
1. Name the collection or category union before fetching it. For a relationship task, identify the complete source of relationship keys and the participant field to compare. For a category union, include every documented source that can contain the requested kind of member.
2. Read the detailed endpoint documentation before relying on a list result. Record the default and maximum page size, page index semantics, sort order, supported filters, date field, and any documented terminal condition. Do not treat a default result window as exhaustive.
3. For a relative date window, observe the current date explicitly. Compute both inclusive boundaries from that observation, print them, and filter on the documented date field. Never infer today or the previous calendar period from the newest returned record.
4. When an exhaustive set is required, request the largest documented page size and batch related page reads in one execute block when practical. Continue only until one of these definitive conditions holds: an empty page, a short page when the maximum page size was used, a documented terminal condition, a filter boundary that excludes the remaining possible records, or the unique requested target is resolved.
5. Preserve decision-critical identifiers and fields needed for the exact predicate. Build sets or unions explicitly; do not guess an entity, identifier, or category that was not returned. Deduplicate only when the task asks for unique members.
6. Before a side effect or completion, verify the selected set against the stated predicate and boundaries. Confirm that every required source was considered and that no extra member from an unrequested relationship, category, or date was included. If the evidence is incomplete, fetch the missing page or category rather than proceeding on an assumption.
7. Stop as soon as the boundary is definitive or the unique target is resolved. Do not force additional discovery after a short page, a documented terminal condition, a definitive filter boundary, or a resolved target. Preserve successful targeted-query procedures when they already establish the latest relevant reply.

Completion check: report or act on only the exact selected set. If a required source or boundary cannot be established with the available APIs, state that limitation in the task result rather than inventing completeness.
