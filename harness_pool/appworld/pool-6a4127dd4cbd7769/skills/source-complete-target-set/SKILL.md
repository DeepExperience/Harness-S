---
name: source-complete-target-set
description: "Derive complete, source-backed target sets before side effects or answers."
whenToUse: "Use when a task requires all items from one or more collections, an unresolved task-critical value, or selection under multiple predicates, relationships, or exceptions. Do not use it when every required identifier and field is already resolved."
disable-model-invocation: false
user-invocable: false
---

Procedure:

1. Identify every source needed to derive the operative target set. Read the relevant API documentation and use its parameters exactly; do not invent query fields or treat a response-schema example as runtime data.
2. For a collection source, inspect the documented pagination contract before selecting records. If a cursor, page index, or equivalent is documented, traverse the collection until a page is shorter than the documented page size or empty. Batch the bounded traversal loop with related reads inside one environment execute call, preserve page state, and do not repeat a page.
3. Treat each task-critical identifier, amount, code, path, recipient, or other value as unresolved until it is observed in the field of a named source object or a directly relevant authoritative response. If a lookup fails, correct the call from its schema and retry; never substitute a guess, placeholder, example value, or superficially related item.
4. Extract every predicate, relationship, exception, and boundary condition from the request and source observations. Represent candidate sets explicitly and intersect all required conditions before any side effect. Deduplicate stable identifiers when combining sources.
5. Verify that the final target set contains exactly the items justified by the complete evidence, perform the requested operation on that set, and verify the resulting state when a documented read is available. Stop once the requested outcome is verified; do not explore unrelated sources or paginate a collection that is already complete.
