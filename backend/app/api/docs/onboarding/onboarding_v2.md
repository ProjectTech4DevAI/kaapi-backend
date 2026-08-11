# Onboarding API v2

Same behavior as v1 onboarding (organization / project / user handling, transactional guarantee), with these differences:

- The vanilla `google` provider is **rejected** — pin the Gemini backend explicitly with `google-aistudio` or `google-gcp`. Existing `google` credential rows keep working at runtime.
- The request body is capped at **32 KB**; larger payloads return **413**.

### Supported Providers
- **LLM:** openai, google-aistudio, google-gcp, anthropic, sarvamai
- **Observability:** langfuse
- **Audio:** elevenlabs

`google-gcp` (Gemini on Vertex AI) requires all of: `api_key`, `project_id`, `location`, `sa_key` (the service-account key JSON as an object), `gcs_bucket`.

### Example
```json
{
  "organization_name": "Acme Foundation",
  "project_name": "acme-field-surveys",
  "credentials": [
    {
      "google-gcp": {
        "api_key": "AQ.Ab8...",
        "project_id": "my-gcp-project",
        "location": "us-central1",
        "sa_key": {
          "type": "service_account",
          "project_id": "my-gcp-project",
          "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
          "client_email": "svc@my-gcp-project.iam.gserviceaccount.com"
        },
        "gcs_bucket": "my-audio-staging-bucket"
      }
    },
    {
      "google-aistudio": {
        "api_key": "AIzaSy..."
      }
    }
  ]
}
```
