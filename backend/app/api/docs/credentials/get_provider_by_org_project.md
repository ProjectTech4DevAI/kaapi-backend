Get credentials for a specific provider within an organization and project.

Retrieves credentials for a specific provider (e.g., `openai`, `langfuse`) for the specified organization and project. Sensitive fields (e.g., `api_key`, `secret_key`) are masked in the response. If credentials for the provider are not configured, `null` is returned. Requires superuser access.

### Path Parameters:
- **org_id**: Organization ID
- **project_id**: Project ID
- **provider**: Provider name (e.g., `openai`, `langfuse`, `google`, `sarvamai`, `elevenlabs`)
