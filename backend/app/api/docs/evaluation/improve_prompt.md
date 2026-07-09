Enqueue an AI prompt-improvement job for the configuration that was evaluated by this run.

The heavy LLM round-trip runs on a background worker, so this endpoint validates preconditions and returns immediately with `202 Accepted` and a `job_id`. Poll `GET /evaluations/{evaluation_id}/improve-prompt/{job_id}` for the result. The run must have `status = completed` and a `score_trace_url` pointing to the stored trace file. No request body is required — the worker reads the trace file, asks Claude to identify low-performing answers (by scores and divergence from ground truth) and rewrite the system prompt. Only `completion.params.instructions` is changed; model, knowledge base, and all other settings are preserved.

On success the new prompt is persisted as the next `config_version` for the evaluated configuration, with AI provenance and the source evaluation run id recorded in the version's `commit_message` (prefixed with `[AI Generated]`).

**Validation errors (returned synchronously before the job is created):**
- `404 evaluation_not_found` — no run with this id in the caller's project.
- `409 evaluation_not_completed` — run is not yet completed.
- `409 source_config_unavailable` — the run's config or config_version has been deleted.
- `422 traces_not_available` — the run has no `score_trace_url`; trace file is required.
- `500 prompt_improvement_enqueue_failed` — the job could not be queued.

LLM and trace-download failures surface on the job itself (status `FAILED`, with `error_message`), not on this response.
