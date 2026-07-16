Start a v2 evaluation run. A full replica of the v1 `POST /api/v1/evaluations`
trigger with Kaapi's native LLM-as-Judge built in — v1 is left unchanged.

In `fast` run mode every scoreable row is automatically judged (no opt-in flag)
on **Adherence to Ground Truth**: an LLM judge scores whether the answer conveys
the same correct information as the dataset's golden answer (0–1, with reasoning),
alongside the existing cosine similarity. Scores, the durable per-row map
(`per_item_ground_truth`), and the `ground_truth_judge` cost stage are all stored
natively by Kaapi. v2 runs do **not** sync to Langfuse.

`batch` run mode mirrors the v1 batch path and is not judged in this phase.

Judging is system-config only: the judge always uses the fallback model
(`gpt-5-mini`) and the built-in ground-truth prompt. There is no per-run or ad-hoc
judge configuration.

## Example (fast)

```json
{
  "dataset_id": 123,
  "experiment_name": "judge-smoke-1",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1,
  "run_mode": "fast"
}
```

## Error responses

| Status | Code | When |
| --- | --- | --- |
| 409 | `run_name_already_exists` | A run with the same `experiment_name` already exists for this (organization, project) |
| 422 | — | In fast mode, an unsupported config type / oversized dataset |
