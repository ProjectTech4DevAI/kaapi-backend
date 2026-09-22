Enqueue a v2 prompt-recommendation job for the configuration evaluated by a judged run.

Unlike v1 (which reads cosine similarity + correctness), this consumes the native
three-metric judge results — **Adherence to Ground Truth**, **Adherence to Prompt**,
and **Adherence to Knowledge Base** — each carrying an integer 0–5 score and the judge's
reasoning. The worker reads both the score and the reasoning per metric, focuses the
rewrite on rows where Adherence to Prompt / Ground Truth are low, and changes only
`completion.params.instructions`; model, knowledge base, and all other settings are
preserved. Low Adherence to Knowledge Base is treated as a retrieval/KB gap rather
than a prompt fault.

The run must be a **judged v2 run** (`is_judge_run = true`), `status = completed`,
and have a `score_trace_url`. The request body is identical to v1 — `callback_url`
only. The LLM round-trip runs on a background worker, so this returns `202 Accepted`
with a `job_id` immediately.

The success/failure result is POSTed to `callback_url` as an `APIResponse` envelope
whose `data` is a `PromptRecommendationJobPublic`. It adds `recommendation_type`,
which is `"prompt"` for now; knowledge-base and model recommendation types are
deferred to a later phase but the field keeps the callback API extensible.

**Validation errors (returned synchronously before the job is created):**
- `404 evaluation_not_found` — no run with this id in the caller's project.
- `409 evaluation_not_completed` — run is not yet completed.
- `409 source_config_unavailable` — the run's config or config_version has been deleted.
- `422 traces_not_available` — the run has no `score_trace_url`.
- `422 not_a_judge_run` — the run is not a judged (v2) run.
- `422 invalid_callback_url` — `callback_url` is missing, not `https://`, or resolves to a blocked (private/loopback/metadata) address.
- `500 prompt_improvement_enqueue_failed` — the job could not be queued.
