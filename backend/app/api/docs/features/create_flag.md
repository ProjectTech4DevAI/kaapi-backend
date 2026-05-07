Create a project-scoped feature flag. Superuser only.

Required payload fields:
- `key`
- `organization_id`
- `project_id`
- `enabled`
- Currently supported key(s): `ASSESSMENT`.

Validation:
- `organization_id` must exist.
- `project_id` must exist and belong to `organization_id`.

Behavior:
- A flag is unique by (`key`, `organization_id`, `project_id`).
- Returns `409` if the same flag already exists.
