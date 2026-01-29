Get the current status and results of a specific evaluation run by the evaluation ID along with some optional query parameters listed below.

Returns comprehensive evaluation information including processing status, configuration, progress metrics, and detailed scores with Q&A context when requested. You can check this endpoint periodically to get to know the evaluation progress. Evaluations are processed asynchronously with status checks every 60 seconds.

**Query Parameters:**
* `get_trace_info` (optional, default: false) - Include Langfuse trace scores with Q&A context. Data is fetched from Langfuse on first request and cached for subsequent calls. Only available for completed evaluations.
* `resync_score` (optional, default: false) - Clear cached scores and re-fetch from Langfuse. Useful when evaluators have been updated. Requires `get_trace_info=true`.
* `export_format` (optional, default: row) -  Controls the structure of traces in the response. Requires `get_trace_info=true` when set to "grouped". Allowed values: `row`, `grouped`.

**Score Format** (`get_trace_info=true`,`export_format=row`):

```json
{
  "summary_scores": [
    {
      "name": "cosine_similarity",
      "avg": 0.87,
      "std": 0.12,
      "total_pairs": 50,
      "data_type": "NUMERIC"
    },
    {
      "name": "response_category",
      "distribution": {"CORRECT": 10, "PARTIAL": 5, "INCORRECT": 2},
      "total_pairs": 17,
      "data_type": "CATEGORICAL"
    }
  ],
  "traces": [
    {
      "trace_id": "uuid-123",
      "question": "What is 2+2?",
      "llm_answer": "4",
      "ground_truth_answer": "4",
      "scores": [
        {
          "name": "cosine_similarity",
          "value": 0.95,
          "data_type": "NUMERIC"
        },
        {
          "name": "correctness",
          "value": 1,
          "data_type": "NUMERIC",
          "comment": "Response is correct"
        }
      ]
    }
  ]
}
```

**Score Format** (`get_trace_info=true`,`export_format=grouped`):
```json
{
  "summary_scores": [...],
  "traces": [...],
  "grouped_traces": [
    {
      "question_id": 1,
      "question": "What is Python?",
      "ground_truth_answer": "Python is a high-level programming language.",
      "llm_answers": [
        "Answer from evaluation run 1...",
        "Answer from evaluation run 2..."
      ],
      "trace_ids": [
        "uuid-123",
        "uuid-456"
      ],
      "scores": [
        [{"name": "cosine_similarity", "value": 0.82, "data_type": "NUMERIC"}],
        [{"name": "cosine_similarity", "value": 0.75, "data_type": "NUMERIC"}]
      ]
    }
  ]
}
```

**Score Details:**
* NUMERIC scores include average (`avg`) and standard deviation (`std`) in summary
* CATEGORICAL scores include distribution counts in summary
* Only complete scores are included (all traces have been rated)
* Numeric values are rounded to 2 decimal places
