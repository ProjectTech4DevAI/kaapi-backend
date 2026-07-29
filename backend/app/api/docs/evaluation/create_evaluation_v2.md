Start a v2 evaluation run. Replicates the v1 `POST /api/v1/evaluations` request
body with Kaapi's native LLM-as-Judge built in — v1 is left unchanged.

v2 runs are **always fast** and always judged (there is no `run_mode`; batch is
deferred to a later phase). Every scoreable row is automatically judged (no opt-in
flag) by one combined LLM-judge call scoring each applicable metric in [0, 1] with
reasoning: **Adherence to Ground Truth** (answer conveys the same correct
information as the golden answer), **Adherence to Prompt** (answer obeys the
assistant's configured instructions; applies only when the run resolves a config
prompt), and **Adherence to Knowledge Base** (groundedness of the answer against
the retrieved chunks; applies only to rows that retrieved chunks). Per-row scores +
reasoning are stored natively by Kaapi in the `score_trace_url` trace unit; the
single `judge` cost stage covers the one combined call. v2 runs compute **no cosine
similarity** and do **not** touch Langfuse.

Judging is system-config only: the judge always uses the configured model
(`EVAL_JUDGE_MODEL`, default `gpt-5-mini`) and the built-in per-metric prompts.
There is no per-run or ad-hoc judge configuration.

## Example

```json
{
  "dataset_id": 123,
  "experiment_name": "judge-smoke-1",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1
}
```

## Error responses

| Status | Code | When |
| --- | --- | --- |
| 409 | `run_name_already_exists` | A run with the same `experiment_name` already exists for this (organization, project) |
| 422 | — | An unsupported config type / oversized dataset |
