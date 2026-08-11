Persist new credentials for the current organization and project.

Credentials are encrypted and stored securely for provider integrations (OpenAI, Langfuse, etc.). Only one credential per provider is allowed per organization-project combination. You can send credentials for a single provider or multiple providers in one request. Refer to the examples below for the required input parameters for each provider.

### Supported Providers:
- **LLM:** openai, anthropic, sarvamai, google(gemini, v1 only — deprecated), google-aistudio, google-gcp (v2 only)
- **Observability:** langfuse
- **Audio:** elevenlabs
- **Miscellaneous** webhook_secret

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
    "anthropic": {
      "api_key": "sk-ant-..."
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
    },
    "webhook_secret": {
      "webhook_secret": "webhook_secret"
    }
  }
}
```
#### For registering Webhook Secret
```json
{
  "credential": {
    "webhook_secret": {
      "webhook_secret": "webhook_secret"
    }
  }
}
```
