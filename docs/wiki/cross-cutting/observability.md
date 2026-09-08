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
- GenAI content never reaches Sentry, in either PII mode. Four locks in `core/sentry_filters.py` + both init sites: `scrub_genai_content` strips `_GENAI_CONTENT_KEYS` (prompt/completion span+log attributes) from every event, `before_send_log_filter` does the same for logs, `genai_privacy_integrations()` pins Sentry's auto-enabled AI integrations to `include_prompts=False`, and `include_local_variables=False` / `max_request_body_size="never"` keep prompts out of stack frames and request bodies. Langfuse (isolated tracer provider, `core/langfuse/langfuse.py`) stays the only sink for message bodies.
- Corollary for log lines on LLM paths: log lengths/ids/types, never prompt or response text — `enable_logs=True` ships every INFO record to Sentry.
- Release: `resolve_sentry_release()` in `core/telemetry.py` (`SENTRY_RELEASE` else `<service>@<API_VERSION>`).
- Sampling/profiling/PII: `SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_PROFILE_SESSION_SAMPLE_RATE`, `SENTRY_PROFILE_LIFECYCLE`, `SENTRY_SEND_DEFAULT_PII`, `SENTRY_ERROR_SAMPLE_RATE` (config.py; defaults preserve current behavior).
- Trace propagation: `CeleryIntegration(propagate_traces=True)` links API to worker as one trace; poll-loop re-enqueues pass `SENTRY_NO_PROPAGATE_HEADERS` (`celery/tasks/job_execution.py`) to start a fresh trace.
- Error capture: generic handler in `core/exception_handlers.py` calls `capture_exception`.
- Tenant impact (both in `core/telemetry.py`, called from `api/deps.py`): `set_request_log_context` puts org/project in the log context + Sentry tags; `bind_sentry_user` binds user/org/project via `sentry_sdk.set_user` so issues report users/orgs affected. The user binding is scope-only — ids in the log context would stamp per-user cardinality on every INFO record `enable_logs` ships. Per-request identity is the `correlation_id` tag (`core/middleware.py`), not the user binding.
- Crons: `@sentry_sdk.monitor` on `/cron/*` endpoints (`api/routes/cron.py`).
- Runbook (alerts/dashboards/debugging): `features/sentry-utilization/SENTRY-RUNBOOK.md`.

## Telemetry / misc
- `core/telemetry.py` — OTel setup, span noise filter, DB/HTTP/LLM metrics, DB spans (`db.statement`, `db.rows_affected`), continuous profiling constants.
- OTel auto-instrumentation (in `setup_telemetry`): FastAPI, SQLAlchemy, httpx, requests, logging, Celery (Queues insight), Redis (Caches insight), botocore (S3/KMS spans).
- `core/rate_monitor.py` — provider rate tracking
