List organizations. **Requires superuser access.**

Returns a paginated list of organizations. The response includes a `has_more` field in `metadata` indicating whether additional pages are available.

**Query Parameters:**
- `search` (optional): Case-insensitive substring match on the organization **name**. For example, `?search=acme` returns every organization whose name contains "acme".
- `is_active` (optional, default `true`): Filter by active status. Pass `false` to list soft-deleted organizations.
- `skip` (optional, default `0`): Number of records to skip for pagination.
- `limit` (optional, default `100`, max `100`): Maximum number of records to return.
