List the validator configs belonging to the calling project, oldest first.

### Query parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `ids` | UUID | no | — | Repeat the parameter to fetch several by id (`?ids=A&ids=B`). |
| `stage` | enum | no | — | `input` or `output`. |
| `type` | enum | no | — | A validator type from `GET /guardrails`. |

This route is not paginated.

### Response

Each item is the stored row flattened together with its validator-specific tuning config, so entries carry keys beyond the declared schema — `id`, `organization_id`, `project_id`, `created_at`, `updated_at`, and every tuning key for that validator type.

### Errors

- `422` — a value in `ids` is not a valid UUID, or `stage`/`type` is not a recognised value.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
