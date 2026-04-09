Add one or more users to a project by email. **Requires superuser access.**

**Request Body:**
- `organization_id` (required): The ID of the organization the project belongs to.
- `project_id` (required): The ID of the project to add users to.
- `users` (required): Array of user objects.
  - `email` (required): User's email address.
  - `full_name` (optional): User's full name.

**Examples:**
- **Single user**: `{"organization_id": 1, "project_id": 1, "users": [{"email": "user@gmail.com", "full_name": "User Name"}]}`
- **Multiple users**: `{"organization_id": 1, "project_id": 1, "users": [{"email": "a@gmail.com"}, {"email": "b@gmail.com"}]}`

**Behavior per email:**
- If the user does not exist, a new account is created with `is_active: false`. The user will be activated on their first Google login.
- If the user already exists and is already in this project, they are skipped.
- If the user exists but is not in this project, they are added.
