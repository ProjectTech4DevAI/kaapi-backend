Start an evaluation run against a stored dataset.

Two execution modes are supported via the optional `run_mode` field:

* `batch` (default) — submits the work to the OpenAI Batch API. Cost-efficient
  for large datasets; turnaround can take up to 24 hours.
* `fast` — runs the evaluation synchronously through the OpenAI Responses API
  and returns results within seconds-to-minutes. Restricted to text
  evaluations on datasets with at most `EVAL_FAST_MAX_UNIQUE_ROWS` unique rows.

**Key Features:**
* Fetches dataset items from Langfuse and creates a job (batch or fast)
* Uses a stored config (created via `/configs`) to define the provider parameters
* Same scoring semantics across both modes — cosine similarity, Langfuse traces,
  and optional LLM-as-Judge correctness
* Use `GET /evaluations/{evaluation_id}` to monitor progress and retrieve results;
  the response carries `run_mode` so clients can tell the two paths apart

## Example (batch — default)

```json
{
  "dataset_id": 123,
  "experiment_name": "gpt4_file_search_test",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1
}
```

## Example (fast)

```json
{
  "dataset_id": 123,
  "experiment_name": "may19-temp0.2-gpt4o-fast",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1,
  "run_mode": "fast"
}
```

## Fast-mode error responses

These apply only when `run_mode` is `fast`.

| Status | Code | When |
| --- | --- | --- |
| 422 | `config_type_unsupported` | Resolved config is not a text-evaluation config |
| 422 | `dataset_too_large_for_fast` | Dataset exceeds `EVAL_FAST_MAX_UNIQUE_ROWS` unique rows |

## General error responses

These apply to both `batch` and `fast` modes.

| Status | Code | When |
| --- | --- | --- |
| 409 | `run_name_already_exists` | A run with the same `experiment_name` already exists for this (organization, project) |
