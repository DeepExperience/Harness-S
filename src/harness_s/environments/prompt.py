"""Fixed AppWorld task contract used by the experiments."""
from typing import Any

def task_prompt(task: dict[str, Any]) -> str:
    supervisor = task.get("supervisor") or {}
    return f"""Solve this AppWorld task through `mcp__environment__execute`. If the Harness exposes the native `skill` tool, you may load its reusable instructions; it is not an AppWorld action and must never replace execution through `mcp__environment__execute`.

Supervisor: {supervisor.get('first_name', '')} {supervisor.get('last_name', '')}
Email: {supervisor.get('email', '')}
Phone: {supervisor.get('phone_number', '')}
Task: {str(task['instruction']).strip()}

The tool is a persistent Python REPL with an `apis` object. Discover APIs with:
- `print(apis.api_docs.show_app_descriptions())`
- `print(apis.api_docs.show_api_descriptions(app_name='...'))`
- `print(apis.api_docs.show_api_doc(app_name='...', api_name='...'))`

Follow this fixed AppWorld execution contract. Before every execute call,
reject your draft if it imports or replaces `apis`:

1. `apis` is already available in the REPL. Never `import apis`, instantiate
   another API collection, or use `apis.apps`; call `apis.<app>.<api>` directly.
2. Inspect the exact API documentation before an unfamiliar call and follow
   its parameter and response schema. If the relevant app is already clear
   from the task, you do not need to list every available app first.
3. Account passwords come from `apis.supervisor.show_account_passwords()`.
   Read them once and build a mapping such as
   `passwords = {{x['account_name']: x['password'] for x in
   apis.supervisor.show_account_passwords()}}`. Log in once per needed app,
   save each returned `access_token`, and pass that token—not the password—to
   every protected call. In AppWorld, only the `phone` app uses the supervisor
   Phone shown above as its login username; every other app uses the supervisor
   Email shown above. For example:
   `login = apis.phone.login(username='<Phone above>',
   password=passwords['phone']); phone_token = login['access_token']`, then
   call `apis.phone.<api>(access_token=phone_token, ...)`. Never omit the saved
   token from a documented protected call and never substitute the password.
4. One `execute` call is one Python code block, not one API call. A block may
   contain several related documented reads, loops, pagination, filtering, or
   writes. Batch safe related work and print only the concise state needed for
   the next decision so that the 30-execution budget is not wasted.
5. Paginated APIs must be exhausted when the task says "all" or otherwise
   depends on the complete collection. References to files mean the
   `file_system` app. Personal relationships such as friends or family are
   resolved through the `phone` app. For date/time requests, compute explicit
   inclusive boundaries from the task environment's current time and default
   time zone. Do not use OS, package, network, task specs, ground truth, or
   evaluator side channels.
6. Treat errors as state. On 401, verify the documented login identifier and
   the corresponding saved access token. On 422, reopen the exact API doc and
   correct parameter names, types, and required fields. Never use an output
   variable from a failed call or repeat an identical failing call.
7. AppWorld is complete only after calling
   `apis.supervisor.complete_task()`. For an information-seeking task, call
   `apis.supervisor.complete_task(answer=<concise_entity_list_or_number>)`;
   for a side-effect task, call it without `answer`. If genuinely unsolvable,
   call `apis.supervisor.complete_task(status="fail")`; a natural-language final response does not
   complete the task. Reserve an execution for this
   mandatory final call and do it as soon as the result is achieved.

Variables persist between calls. Use documented app APIs to finish the task,
then finish through `apis.supervisor.complete_task(...)` as described above."""
