Start a v2 evaluation run. Replicates the v1 `POST /api/v1/evaluations` request
body with Kaapi's native LLM-as-Judge built in — v1 is left unchanged.

v2 runs are **always fast** and always judged (there is no `run_mode`; batch is
deferred to a later phase). Every scoreable row is automatically judged (no opt-in
flag) on **Adherence to Ground Truth**: an LLM judge scores whether the answer
conveys the same correct information as the dataset's golden answer (0–1, with
reasoning). Scores, the durable per-row map (`per_item_ground_truth`), and the
`ground_truth_judge` cost stage are stored natively by Kaapi. v2 runs compute **no
cosine similarity** and do **not** touch Langfuse.

Judging is system-config only: the judge always uses the configured model
(`EVAL_JUDGE_MODEL`, default `gpt-5-mini`) and the built-in ground-truth prompt.
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
