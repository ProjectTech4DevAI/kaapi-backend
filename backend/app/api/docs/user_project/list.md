List all users that belong to a project.

**Query Parameters:**
- `project_id` (required): The ID of the project to list users for.

Returns user details including their active status — users added via invitation will have `is_active: false` until they complete their first Google login.
