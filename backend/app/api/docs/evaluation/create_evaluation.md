Start an evaluation run using the OpenAI Batch API.

Evaluations allow you to systematically test LLM configurations against
predefined datasets with automatic progress tracking and result collection.

**Key Features:**
* Fetches dataset items from Langfuse and creates a batch processing job via the OpenAI Batch API
* Asynchronous processing with automatic progress tracking (checks every 60s)
* Uses a stored config (created via `/configs`) to define the provider parameters
* Stores results for comparison and analysis
* Use `GET /evaluations/{evaluation_id}` to monitor progress and retrieve results

## Example

```json
{
  "dataset_id": 123,
  "experiment_name": "gpt4_file_search_test",
  "config_id": "f54f0d67-4817-4103-9fdf-b74b3d46733e",
  "config_version": 1
}
```
