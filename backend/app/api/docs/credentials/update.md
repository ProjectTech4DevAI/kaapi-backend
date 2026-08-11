Update credentials for a specific provider.

Updates existing provider credentials for the current organization and project. If the credentials for the specified provider don't exist yet, they will be **created** automatically (upsert behavior). The `provider` and `credential` fields are required.

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
- **LLM:** openai, sarvamai, google(gemini, v1 only — deprecated), google-aistudio, google-gcp (v2 only)
- **Observability:** langfuse
- **Audio:** elevenlabs
- **Miscellaneous** webhook_secret
