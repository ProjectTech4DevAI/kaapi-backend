Start an evaluation run using the OpenAI Batch API.

Evaluations allow you to systematically test LLM configurations against
predefined datasets with automatic progress tracking and result collection.

**Key Features:**
* Fetches dataset items from Langfuse and creates batch processing job via OpenAI Batch API
* Asynchronous processing with automatic progress tracking (checks every 60s)
* Supports configuration from direct parameters or existing assistants
* Stores results for comparison and analysis
* Use `GET /evaluations/{evaluation_id}` to monitor progress and retrieve results of evaluation.

**Example: Using Direct Configuration**

```json
{
  "dataset_id": 123,
  "experiment_name": "gpt4_file_search_test",
  "config": {
    "model": "gpt-4o",
    "instructions": "You are a helpful FAQ assistant for farmers.",
    "tools": [
      {
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"],
        "max_num_results": 5
      }
    ],
    "temperature": 0.7,
    "include": ["file_search_call.results"]
  }
}
```

**Example: Using Existing Assistant**

```json
{
  "dataset_id": 123,
  "experiment_name": "production_assistant_eval",
  "config": {},
  "assistant_id": "asst_xyz789"
}
```

**Note:** When using `assistant_id`, configuration is fetched from the assistant in the database. You can pass `config` as an empty object `{}`.
