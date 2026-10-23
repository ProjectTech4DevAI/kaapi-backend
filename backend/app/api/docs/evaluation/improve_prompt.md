Enqueue an AI prompt-improvement job for the configuration that was evaluated by this run.

The heavy LLM round-trip runs on a background worker, so this endpoint validates preconditions and returns immediately with `202 Accepted` and a `job_id`. The run must have `status = completed` and a `score_trace_url` pointing to the stored trace file. The worker reads the trace file, asks Claude to identify low-performing answers (by scores and divergence from ground truth) and rewrite the system prompt. Only `completion.params.instructions` is changed; model, knowledge base, and all other settings are preserved.

On success the new prompt is persisted as the next `config_version` for the evaluated configuration, with AI provenance and the source evaluation run id recorded in the version's `commit_message` (prefixed with `[AI Generated]`).

**Request body (required):**

```json
{ "callback_url": "https://your-service.example.com/webhooks/prompt-improvement" }
```

- `callback_url` — HTTPS webhook that receives the result. Must be `https://` and must not resolve to a private, loopback, or cloud-metadata address.

**Callback delivery:**

Once the worker finishes (success or failure), it POSTs a single JSON body to `callback_url`. Delivery is **best-effort, single-attempt** — there is no retry. The `config_version` is persisted regardless of callback outcome and remains recoverable from the configuration's version list, so a failed callback loses the notification, not the work.

When a webhook signing secret is configured for the project, the request carries `X-Webhook-Signature` (HMAC-SHA256 of the body) and `X-Webhook-Timestamp` headers; verify them before trusting the payload.

The body is a standard `APIResponse` envelope whose `data` is a `PromptImprovementJobPublic`:

```json
{
  "success": true,
  "data": {
    "job_id": "…",
    "status": "SUCCESS",
    "config_version": { "…": "…" },
    "error_message": null
  },
  "error": null,
  "metadata": null
}
```

```json
{
  "success": false,
  "data": {
    "job_id": "…",
    "status": "FAILED",
    "config_version": null,
    "error_message": "…"
  },
  "error": "…",
  "metadata": null
}
```

LLM and trace-download failures surface only via the failure callback, not on the `202` response.

**Validation errors (returned synchronously before the job is created):**
- `404 evaluation_not_found` — no run with this id in the caller's project.
- `409 evaluation_not_completed` — run is not yet completed.
- `409 source_config_unavailable` — the run's config or config_version has been deleted.
- `422 traces_not_available` — the run has no `score_trace_url`; trace file is required.
- `422 invalid_callback_url` — `callback_url` is missing, not `https://`, or resolves to a blocked (private/loopback/metadata) address.
- `500 prompt_improvement_enqueue_failed` — the job could not be queued.
