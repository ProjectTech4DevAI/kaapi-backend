Generate a new, AI-improved prompt iteration for the configuration that was evaluated by this run.

The run must have `status = completed` and a `score_trace_url` pointing to the stored trace file. No request body is required — the service reads the trace file directly from S3, uploads it to the Anthropic Files API, and asks Claude to identify low-performing answers (by scores and divergence from ground truth) and rewrite the system prompt to address them. Only `completion.params.instructions` is changed — model, knowledge base, and all other settings are preserved.

The new prompt is persisted as the next `config_version` for the evaluated configuration. AI provenance and the source evaluation run id are recorded in the version's `commit_message` field (prefixed with `[AI Generated]` for auditability).

**Error codes:**
- `404 evaluation_not_found` — no run with this id in the caller's project.
- `409 evaluation_not_completed` — run is not yet completed.
- `409 source_config_unavailable` — the run's config or config_version has been deleted.
- `422 traces_not_available` — the run has no `score_trace_url`; trace file is required.
- `502 trace_download_failed` — the trace file could not be retrieved from storage.
- `502 prompt_generation_failed` — the platform Anthropic key is not configured, or the LLM call failed.
