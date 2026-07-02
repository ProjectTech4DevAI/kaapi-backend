List projects for a given organization. **Requires superuser access.**

Returns the projects belonging to the specified organization ID. The organization must exist and be active.

**Query Parameters:**
- `search` (optional): Case-insensitive substring match on the project **name**. For example, `?search=onboarding` returns only projects whose name contains "onboarding".
- `is_active` (optional, default `true`): Filter by active status. Pass `false` to list soft-deleted projects (e.g. to selectively reactivate them).
