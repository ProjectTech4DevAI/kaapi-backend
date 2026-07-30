# Module: Platform (misc shared surfaces)

Small shared surfaces that don't warrant their own page.

All paths relative to `backend/app/`.

| Surface | Routes | Tables / Models | Logic |
|---|---|---|---|
| Analytics | `api/routes/analytics.py` | `models/analytics.py` | `services/analytics/` |
| Notifications | — | `notification` (`models/notification.py`) | `services/notifications/`, `crud/notification.py` |
| Feature flags | `api/routes/features.py` | `feature_flag` (`models/feature_flag.py`) | `core/feature_flags/`, `crud/feature_flag.py` |
| Languages | `api/routes/languages.py` | `global.languages` (`models/language.py`) | `crud/language.py` |
| Credentials | `api/routes/credentials.py` | `credential` (`models/credentials.py`) | `crud/credentials.py`; provider keys per org/project |
| Model config | `api/routes/model_config.py` | `model_config` (`models/model_config.py`) | `crud/model_config.py` |
| Cron | `api/routes/cron.py` | — | triggers batch polling (`crud/evaluations/cron.py`); health probes (`services/health_probes.py`) fire one round-robin probe per tick through the real `/llm/call` pipeline (`services/llm/jobs.py::start_job`), rotation state in Redis (`health_probe:index`, `health_probe:last_job_id`), no dedicated table — result rides the `job` table |
| Jobs | — | `job` (`models/job.py`), `batch_job` (`models/batch_job.py`) | `crud/jobs.py`, `crud/job/`, `services/job_monitoring.py` |
