Generate a new, AI-improved prompt iteration for the configuration that was evaluated by this run.

The run must have `status = completed`. Supply a `metric` name (the exact score name as it appears in the run's `summary_scores`, matched case-insensitively) and a numeric `threshold`. The service identifies questions that scored below `threshold` consistently across their repetitions, and categories whose mean score is below `threshold`, then asks Claude to rewrite the prompt to target those weaknesses. Only `completion.params.instructions` is changed — model, knowledge base, and all other settings are preserved byte-for-byte.

**Dynamic metric name:** `metric` is a free-form string, not a fixed enum. Any NUMERIC score recorded in the run's `summary_scores` is valid (e.g. `"Cosine Similarity"`, or any Langfuse scorer you have configured). Only NUMERIC scores are supported; passing a CATEGORICAL score name returns `422 metric_not_numeric` (categorical support is Phase 2).

**Threshold:** no range constraint — supply whatever value makes sense for the score's scale (0–1 for cosine similarity, 1–5 for a Likert scorer, 0–100 for a percentage scorer, etc.).

The new prompt is persisted as the next `config_version` for the evaluated configuration. AI provenance, the source evaluation run id, and the metric + threshold used are recorded in the version's `commit_message` field (prefixed with `[AI Generated]` for auditability).

**Error codes:**
- `404 evaluation_not_found` — no run with this id in the caller's project.
- `409 evaluation_not_completed` — run is not yet completed.
- `409 source_config_unavailable` — the run's config or config_version has been deleted.
- `422 metric_not_available` — no summary score with the supplied name exists in this run.
- `422 metric_not_numeric` — the matched score has a non-NUMERIC data type (categorical scores are not yet supported).
- `422 no_weak_signals` — no consistently-low questions and no underperforming categories found; no version is created.
- `502 prompt_generation_failed` — the platform Anthropic key is not configured, or the LLM call failed.
