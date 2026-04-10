Update credentials for a specific provider within an organization and project.

Updates existing provider credentials for the specified organization and project. Provider and credential fields must be provided in the request body. Requires superuser access.

### Path Parameters:
- **org_id**: Organization ID
- **project_id**: Project ID

### Example:
```json
{
  "provider": "openai",
  "credential": {
    "api_key": "sk-proj-..."
  }
}
```
