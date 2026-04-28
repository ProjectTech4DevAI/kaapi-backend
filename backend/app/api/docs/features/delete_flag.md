Delete a project-scoped feature flag. Superuser only.

Required payload fields:
- `key`
- `organization_id`
- `project_id`

Validation:
- `organization_id` must exist.
- `project_id` must exist and belong to `organization_id`.

Returns:
- `{"deleted": true}` on success.
- `404` if the target flag does not exist.
