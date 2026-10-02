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
      "guardrail": null,
      "input_to_llm": null,
      "output_from_llm": null,
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
* `guardrail` (row format only) reports what the config's guardrails did to that row: `"blocked: <reason>"` (no answer was generated, the row is unscoreable with reason `guardrail_blocked` and is not counted as a failure), `"rephrased"` (guardrails answered directly and the answer is scored normally), `"applied"` (guardrails ran and passed the content through), or `null`/absent (guardrails did not run, or were bypassed because the service was unreachable). Fast runs only; batch runs never carry it.
* `input_to_llm` / `output_from_llm` (row format only, fast runs) are set on `"applied"` rows: `input_to_llm` is the prompt the LLM actually received after input guardrails (e.g. with PII redacted, and after `prompt_template` interpolation), `output_from_llm` is the LLM's answer before output guardrails changed it (`llm_answer` stays the post-guardrail text that is scored). Each is `null` when that side's guardrails did not apply, and on blocked or rephrased rows. `output_from_llm` is the pre-redaction text, so it can contain content an output PII guardrail removed.
