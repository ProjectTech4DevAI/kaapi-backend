List feature flag records. Superuser only.

Query parameters are optional:
- `key`
- `organization_id`
- `project_id`

`key` matches stored feature flag values (currently `ASSESSMENT`).

Validation:
- When `project_id` is provided, `organization_id` is required.
- If both are provided, `project_id` must belong to `organization_id`.
