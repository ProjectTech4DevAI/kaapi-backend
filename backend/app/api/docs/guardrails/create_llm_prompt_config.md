Create a stored prompt for one of the LLM-backed validators.

Two validators read their instructions from a stored prompt instead of hard-coding them: `topic_relevance` (does this text stay on topic?) and `answer_relevance_custom_llm` (does this answer address the question?). Create the prompt here, then reference its `id` from the matching validator config.

### Request

```json
{
  "validator_name": "topic_relevance",
  "name": "Maternal Health Scope",
  "description": "Topic guard for maternal health support bot",
  "prompt_schema_version": 1,
  "llm_prompt": "Pregnancy care: Questions about prenatal care, ANC visits, nutrition, supplements, danger signs. Postpartum care: Questions about recovery after delivery, breastfeeding, and mother health checks."
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `validator_name` | enum | yes | `topic_relevance` or `answer_relevance_custom_llm`. Immutable after creation. |
| `name` | string | yes | 1–100 chars. |
| `description` | string | yes | 1–500 chars. |
| `prompt_schema_version` | integer | no | Defaults to `1`. Must be >= 1. |
| `llm_prompt` | string | yes | The prompt text. Non-empty. |

### Placeholders

For `answer_relevance_custom_llm` the prompt **must** contain both `{query}` and `{answer}`; the service rejects it otherwise. Example:

```
You are evaluating a maternal health assistant.
Query: {query}
Answer: {answer}

Does the answer directly address the maternal health query?
Answer only YES or NO.
```

`topic_relevance` prompts have no required placeholders.

### Notes

- New configs are created active. `is_active` can only be changed via `PATCH`.
- Responds `200`, not `201`.

### Errors

- `400` — a config with the same validator, version and prompt text already exists.
- `422` — the body failed validation, or an `answer_relevance_custom_llm` prompt was missing `{query}`/`{answer}`.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
