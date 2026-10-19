Poll a guardrails job for its status and sanitised result.

Use this when you submitted `POST /guardrails` without a `callback_url`, or to inspect a completed job for traceability. The job id comes from the `POST /guardrails` response.

### Status values

| `status` | Meaning |
|---|---|
| `PENDING` | Queued, not yet picked up. |
| `PROCESSING` | A worker is applying the validators. |
| `SUCCESS` | Finished; `guardrails_response` carries the sanitised text. |
| `FAILED` | The text was hard-blocked, or the job errored. `error_message` explains why. |

### Response

`guardrails_response` is populated only on `SUCCESS`; it is `null` in every other state. The sanitised text sits at `guardrails_response.response.output.content.value`.

`warnings` mirrors the `metadata.warnings` of the webhook payload, so polling callers do not miss a bypass signal — most importantly the case where the guardrails service was unavailable and the original text was returned unchanged. It is always empty for a hard-blocked job.

### Notes

- If the upstream response carried no sanitised text, the value falls back to the original submitted text.
- `usage` counters default to zero when the guardrails service reports none.

### Errors

- `404` — no such job in this project, or the id belongs to a job that is not a guardrails job.
- `422` — `job_id` is not a valid UUID.
