---
name: boundary-evidence
description: "Establish completeness for paginated lists and relative date windows before acting."
whenToUse: "Use before treating a list, search result, or relative time window as complete when a decision or write depends on every matching record. Do not use it after a valid boundary is established or when the requested target is already resolved."
disable-model-invocation: false
user-invocable: false
---

1. Read the exact API documentation for the endpoint. Record its default and maximum page size, page-index semantics, filters, sort order, and any documented terminal condition.
2. For a paginated collection, request the largest permitted page size. Continue page by page only until one of: an empty page, a page shorter than the requested limit, a documented terminal condition, or a definitive filter boundary that proves no further relevant record can appear. Preserve decision-critical identifiers and the boundary observation.
3. For a relative date window, obtain the current date and time from an explicit environment observation, such as Python datetime.now() inside execute, rather than the newest record, a notification, or remembered real-world time. Compute the inclusive start and end in the task's date and default time zone, then verify each selected record against those boundaries.
4. Stop as soon as the target is resolved or a valid boundary is established. Do not inspect unrelated categories, re-fetch a valid terminal page, or force discovery after the answer is already complete.
5. If documentation or observations are insufficient to establish a boundary, state that limitation and seek the precise missing evidence rather than treating a bounded result as exhaustive.
