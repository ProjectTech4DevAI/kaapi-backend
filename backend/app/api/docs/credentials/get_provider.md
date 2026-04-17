Get credentials for a specific provider.

Retrieves credentials for a specific provider (e.g., `openai`, `langfuse`) for the current organization and project. Sensitive fields (e.g., `api_key`, `secret_key`) are masked in the response. If credentials for the provider are not configured, `null` is returned.
