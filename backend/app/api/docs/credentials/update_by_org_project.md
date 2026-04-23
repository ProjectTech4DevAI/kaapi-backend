Update credentials for a specific provider within an organization and project.

Updates existing provider credentials for the specified organization and project. If the credentials for the specified provider don't exist yet, they will be **created** automatically (upsert behavior). Requires superuser access.

### Path Parameters:
- **org_id**: Organization ID
- **project_id**: Project ID

The `credential` field accepts **two formats** (both work the same):

### Nested format (same as create endpoint):
```json
{
  "provider": "openai",
  "is_active": true,
  "credential": {
    "openai": {
      "api_key": "sk-proj-..."
    }
  }
}
```

### Flat format:
```json
{
  "provider": "openai",
  "is_active": true,
  "credential": {
    "api_key": "sk-proj-..."
  }
}
```

### Supported Providers:
- **LLM:** openai, sarvamai, google(gemini)
- **Observability:** langfuse
- **Audio:** elevenlabs
- **Miscellaneous** webhook_secret
