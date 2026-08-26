# Cross-cutting: Observability

All paths relative to `backend/app/`.

## Langfuse
- `core/langfuse/langfuse.py` — client integration. LLM calls and evaluation runs write traces; evaluation scores attach to traces (`scores` list, reasoning in score `comment`).
- Durable score maps on `evaluation_run` are the resync source of truth when a Langfuse write fails.

## Logging
- `core/logger.py`. Convention (CLAUDE.md): every log line starts with the function name in brackets — `logger.info(f"[function_name] Message | key: {value}")`.

## Sentry
OTel-first, Sentry as sole in-process sink (`instrumenter="otel"`). Init at `main.py` (web) and `celery/celery_app.py` (worker), kept in sync.
- `core/sentry_filters.py` — `before_send_transaction_filter` (drops probe/low-signal spans) and `before_send_error_filter` (drops probe events, scrubs request PII when `SENTRY_SEND_DEFAULT_PII` off).
- Release: `resolve_sentry_release()` in `core/telemetry.py` (`SENTRY_RELEASE` else `<service>@<API_VERSION>`).
- Sampling/profiling/PII: `SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_PROFILE_SESSION_SAMPLE_RATE`, `SENTRY_PROFILE_LIFECYCLE`, `SENTRY_SEND_DEFAULT_PII`, `SENTRY_ERROR_SAMPLE_RATE` (config.py; defaults preserve current behavior).
- Trace propagation: `CeleryIntegration(propagate_traces=True)` links API to worker as one trace; poll-loop re-enqueues pass `SENTRY_NO_PROPAGATE_HEADERS` (`celery/tasks/job_execution.py`) to start a fresh trace.
- Error capture: generic handler in `core/exception_handlers.py` calls `capture_exception`.
- Tenant impact: `set_request_log_context` (`core/telemetry.py`, called from `api/deps.py`) sets `sentry_sdk.set_user` (user/org/project) so issues report users/orgs affected.
- Crons: `@sentry_sdk.monitor` on `/cron/*` endpoints (`api/routes/cron.py`).
- Runbook (alerts/dashboards/debugging): `features/sentry-utilization/SENTRY-RUNBOOK.md`.

## Telemetry / misc
- `core/telemetry.py` — OTel setup, span noise filter, DB/HTTP/LLM metrics, DB spans (`db.statement`, `db.rows_affected`), continuous profiling constants.
- OTel auto-instrumentation (in `setup_telemetry`): FastAPI, SQLAlchemy, httpx, requests, logging, Celery (Queues insight), Redis (Caches insight), botocore (S3/KMS spans).
- `core/rate_monitor.py` — provider rate tracking
