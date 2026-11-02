---
name: collection-exhaustion-completion
description: "Check paginated collections for explicit exhaustion and match completion answers to task type."
whenToUse: "Use once when a paginated collection must be exhaustive or immediately before complete_task; do not reload it."
disable-model-invocation: false
user-invocable: false
---

1. Use mcp__environment__execute for every AppWorld operation. Never call apis.api_docs or an app API as a separate tool.
2. Decide whether the result will support exhaustive evidence or select mutation targets. If a single targeted match is sufficient, the collection is already explicitly exhausted, or the endpoint documentation shows no pagination, use the result directly.
3. Inside the execute Python block, inspect paginated endpoint documentation for page_index and page_limit. Preserve useful filters and query constraints.
4. Request a bounded larger page within the documented maximum. Continue page_index until an empty page or another explicit exhaustion signal. Do not treat a default-sized first page as complete.
5. Accumulate results across pages and use the complete accumulated set only after exhaustion. Stop once exhaustion is explicit; do not add unnecessary reads.
6. If a call returns 422, inspect the exact API schema inside execute and correct parameter names, types, or required fields before retry; do not repeat identical failing code.
7. Before complete_task, verify the requested state with an ordinary documented read API when needed. Classify the task. For an explicit question, pass only the concise requested answer. For a side-effect-only state change, omit the answer argument. If the task is genuinely unsolvable, use the documented failure status.
