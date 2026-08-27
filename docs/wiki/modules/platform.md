# Module: Platform (misc shared surfaces)

Small shared surfaces that don't warrant their own page.

All paths relative to `backend/app/`.

| Surface | Routes | Tables / Models | Logic |
|---|---|---|---|
| Analytics | `api/routes/analytics.py` | `models/analytics.py` | `services/analytics/` |
| Notifications | — | `notification` (`models/notification.py`) | `services/notifications/`, `crud/notification.py` |
| Feature flags | `api/routes/features.py` | `feature_flag` (`models/feature_flag.py`) | `core/feature_flags/`, `crud/feature_flag.py` |
| Languages | `api/routes/languages.py` | `global.languages` (`models/language.py`) | `crud/language.py` |
| Credentials | `api/routes/credentials.py` | `credential` (`models/credentials.py`) | `crud/credentials.py`; provider keys per org/project; envelope encryption (KMS-wrapped data key + AES-GCM), prefix-versioned ciphertexts |
| Model config | `api/routes/model_config.py` | `model_config` (`models/model_config.py`) | `crud/model_config.py` |
| Bucket providers | — | reuses `credential` (`google-gcp`) | `services/buckets/` — global registry + resolver (`providers/`), GCS V4 signed + bulk-signed URLs with a 24h expiry cap (`providers/gcs.py`, cap in `providers/base.py`), attachment path selection + URL resolution (`attachments.py`) |
| Cron | `api/routes/cron.py` | — | triggers batch polling (`crud/evaluations/cron.py`) |
| Jobs | — | `job` (`models/job.py`), `batch_job` (`models/batch_job.py`) | `crud/jobs.py`, `crud/job/`, `services/job_monitoring.py` |

## Credential contracts

`core/providers.py` holds one Pydantic model per `Provider` (`OpenAICredentials`,
`LangfuseCredentials`, `GoogleCredentials`, …) and `PROVIDER_CONFIGS` maps each
provider to its model plus its sensitive (masked) fields. Those models are the
single source of truth:

- `ProviderConfig.required_fields` is derived from the model, so a schema change
  and the validation gate can't drift apart.
- Request models (`CredsCreate.credential`, `CredsUpdate.credential`,
  `OnboardingRequest.credentials`) are keyed by `Provider` and typed as the union
  of those models, so the shape lands in the OpenAPI schema instead of
  `dict[str, Any]` — adding or changing a required field is a visible contract
  change.
- Payloads allow extra keys (callers store provider knobs beyond the declared
  auth fields); they round-trip via `model_dump(exclude_unset=True)`. A model
  declares the fields Kaapi itself requires — `google` routes to Google AI
  Studio (`LLMProvider._registry`), so it declares `api_key` only.
- Responses (`CredsPublic.credential`, `GET /credentials/provider/{provider}`)
  are masked, so they are typed `CredentialPayload` (`dict[str, JsonValue]`)
  rather than a provider model.

Adding a provider = add the enum member, the model, and its `PROVIDER_CONFIGS`
entry; nothing else needs to know.
