---
name: contract-preflight
description: "Resolve AppWorld API contracts through the environment execute tool before selecting or calling an endpoint, especially when documentation output is silent or a generic operation could replace a relation-specific one."
whenToUse: "Use before first relying on an unfamiliar app or endpoint, after a documentation call renders no useful output, or before choosing between generic and relation-specific operations. Stop once the exact contract and target object are already resolved."
disable-model-invocation: false
user-invocable: false
---

Procedure:
1. Use mcp__environment__execute, not a direct apis.* tool call. Its code argument is Python; for example, print(apis.api_docs.show_app_descriptions()) or print(apis.api_docs.show_api_doc(app_name='...', api_name='...')).
2. Read the printed app-level descriptions and exact endpoint schema. Do not infer documentation is absent from a heading or an unprinted return value.
3. Record supported parameters, required fields, response shape, and pagination. Use only documented filters.
4. If a target is absent from the first page, advance through documented pagination rather than repeating the same call or guessing IDs.
5. Before a side effect, compare generic and relation-specific operations. Choose the operation that matches the source object's scope and lifecycle, such as updating or consuming an existing email, thread, or draft rather than creating a new object.
6. Verify the resulting state with documented reads before completion.
Stop early when the contract and target object are already resolved; do not force rediscovery.
