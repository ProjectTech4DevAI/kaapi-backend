Poll a prompt-improvement job started via `POST /evaluations/{evaluation_id}/improve-prompt`.

Returns the job `status` (`PENDING`, `PROCESSING`, `SUCCESS`, or `FAILED`). When `status = SUCCESS`, `config_version` holds the newly created config version (the AI-improved prompt). When `status = FAILED`, `error_message` describes the failure (missing Anthropic key, LLM error, trace download failure, etc.). Both fields are `null` while the job is still running.

Returns `404` if no job with this id exists in the caller's project.
