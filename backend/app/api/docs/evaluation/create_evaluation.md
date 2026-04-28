Start an evaluation run.

Evaluations allow you to systematically test LLM configurations against
predefined datasets with automatic progress tracking and result collection.

**Execution modes (`run_mode`):**
* `batch` (default) — submits the dataset to the OpenAI Batch API. Best for
  larger datasets; queueing time can be minutes-to-hours but cost is ~50%
  lower per token. The cron poller drives completion.
* `live` — fans out per-row Celery tasks against the regular Responses API.
  Capped at a server-configured item count (`EVAL_LIVE_MAX_ITEMS`, default
  100). Standard (non-batch) pricing applies. Use this for fast feedback on
  small datasets — typically completes in seconds.

**Key Features:**
* Fetches dataset items from Langfuse and runs them through the chosen mode
* Asynchronous processing with automatic progress tracking
* Uses a stored config (created via `/configs`) to define the provider parameters
* Stores results for comparison and analysis
* Use `GET /evaluations/{evaluation_id}` to monitor progress and retrieve results

## Example

```json
{
  "dataset_id": 123,
  "experiment_name": "gpt4_file_search_test",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1,
  "run_mode": "live"
}
```
