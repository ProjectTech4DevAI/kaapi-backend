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
| Bucket providers | — | reuses `credential` (`google-gcp`) | `services/buckets/` — global registry + resolver (`providers/`), GCS signed/bulk-signed/private-to-public URLs (`providers/gcs.py`), attachment path selection + URL resolution (`attachments.py`) |
| Cron | `api/routes/cron.py` | — | triggers batch polling (`crud/evaluations/cron.py`) |
| Jobs | — | `job` (`models/job.py`), `batch_job` (`models/batch_job.py`) | `crud/jobs.py`, `crud/job/`, `services/job_monitoring.py` |
