Persist new credentials for the current organization and project.

Credentials are encrypted and stored securely for provider integrations (OpenAI, Langfuse, etc.). Only one credential per provider is allowed per organization-project combination.

### Supported Providers:
- **LLM:** openai, sarvamai, google(gemini)
- **Observability:** langfuse
- **Audio:** elevenlabs

### Examples:

#### Single Provider
```json
{
  "credential": {
    "openai": {
      "api_key": "sk-proj-..."
    }
  }
}
```

#### Multiple Providers
```json
{
  "credential": {
    "openai": {
      "api_key": "sk-proj-..."
    },
    "google": {
      "api_key": "AIzaSy..."
    },
    "sarvamai": {
      "api_key": "sarvam-..."
    },
    "elevenlabs": {
      "api_key": "sk_..."
    },
    "langfuse": {
      "public_key": "pk-lf-....",
      "secret_key": "sk-lf-...",
      "host": "https://cloud.langfuse.com"
    }
  }
}
```
