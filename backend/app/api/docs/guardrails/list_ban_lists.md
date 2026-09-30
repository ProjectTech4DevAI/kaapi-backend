List the ban lists visible to the calling project.

Returns lists owned by the project plus any list from another tenant marked `is_public`. Ordered newest first.

### Query parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `domain` | string | no | — | Return only lists carrying this domain label. |
| `offset` | integer | no | `0` | Rows to skip. Must be >= 0. |
| `limit` | integer | no | — | Max rows to return, 1–100. Omit for no limit. |

### Errors

- `422` — `offset` is negative, or `limit` is outside 1–100.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
