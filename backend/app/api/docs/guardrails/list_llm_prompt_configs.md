List the stored LLM prompt configs belonging to the calling project, oldest first.

### Query parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `validator_name` | enum | no | — | `topic_relevance` or `answer_relevance_custom_llm`. |
| `offset` | integer | no | `0` | Rows to skip. Must be >= 0. |
| `limit` | integer | no | — | Max rows to return, 1–100. Omit for no limit. |

### Errors

- `422` — `validator_name` is not a recognised value, `offset` is negative, or `limit` is outside 1–100.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
