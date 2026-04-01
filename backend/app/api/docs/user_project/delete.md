Remove a user from a project. **Requires superuser access.**

**Path Parameters:**
- `user_id` (required): The ID of the user to remove.

**Query Parameters:**
- `project_id` (required): The ID of the project to remove the user from.

This only removes the user-project mapping — the user account itself is not deleted. You cannot remove yourself from a project.
