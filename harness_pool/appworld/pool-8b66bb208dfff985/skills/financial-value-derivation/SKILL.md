---
name: financial-value-derivation
description: "Derive exact financial amounts and unpaid participant sets from authoritative app records before payment-platform requests or expense-log writes."
whenToUse: "Use when a task requires a payment request, money transfer, or monthly expense-log update and the amount, unpaid recipient set, or aggregate is not explicitly supplied. Do not use it for read-only financial questions or when the task already gives the exact amount and recipients."
disable-model-invocation: false
user-invocable: false
---

Procedure:
1. Identify the authoritative source app and record type for the required financial value.
2. Retrieve the complete relevant record set with documented filters and pagination; do not rely on a single default page or an unrelated record.
3. Classify each amount as a total, individual share, prior payment, or monthly aggregate. An already-paid individual share is not the total.
4. Compute the required value once from the complete task-matching set and retain it in a named variable.
5. For payment requests, derive the unpaid participant set by reconciling the participant list against authoritative prior payments; do not request from everyone when the task says some have paid.
6. Verify the computed amount, recipient set, description, and any required ordering against the source records before writing.
7. For a monthly expense log, determine the current month from the environment, query the complete source set for that month, sum only task-matching records, and insert the new entry in the log's existing month order.
8. Perform the write only after verification, then read back the changed record.

Stop if the authoritative source is unavailable; report failure rather than substituting a placeholder. If the task already supplies the exact amount and target, proceed directly without this procedure.
