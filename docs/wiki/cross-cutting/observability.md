# Cross-cutting: Observability

All paths relative to `backend/app/`.

## Langfuse
- `core/langfuse/langfuse.py` — client integration. LLM calls and evaluation runs write traces; evaluation scores attach to traces (`scores` list, reasoning in score `comment`).
- Durable score maps on `evaluation_run` are the resync source of truth when a Langfuse write fails.

## Logging
- `core/logger.py`. Convention (CLAUDE.md): every log line starts with the function name in brackets — `logger.info(f"[function_name] Message | key: {value}")`.

## Sentry
OTel-first, Sentry as sole in-process sink (`instrumenter="otel"`). Init at `main.py` (web) and `celery/celery_app.py` (worker), kept in sync.
- `core/telemetry/sentry/filters.py` — `before_send_transaction_filter` (drops probe transactions, scrubs genai content; DB spans are kept so Sentry Queries has data) and `before_send_error_filter` (drops probe events, always drops the request body, scrubs headers/cookies/query when `SENTRY_SEND_DEFAULT_PII` off, redacts LLM task kwargs).
- GenAI content never reaches Sentry, in either PII mode. Four locks in `core/telemetry/sentry/filters.py`, wired by `init_sentry()` in `core/telemetry/sentry/init.py` for both API and worker: `scrub_genai_content` strips `_GENAI_CONTENT_KEYS` (prompt/completion span+log attributes) from every event, `before_send_log_filter` does the same for logs, `genai_privacy_integrations()` pins Sentry's auto-enabled AI integrations to `include_prompts=False`, and `include_local_variables=False` / `max_request_body_size="never"` keep prompts out of stack frames and request bodies. Langfuse (isolated tracer provider, `core/langfuse/langfuse.py`) stays the only sink for message bodies.
- Corollary for log lines on LLM paths: log lengths/ids/types, never prompt or response text — `enable_logs=True` ships every INFO record to Sentry.
- Release: `resolve_sentry_release()` in `core/telemetry/sentry/init.py`: `SENTRY_RELEASE` override, else `<service>@<RELEASE_TAG or API_VERSION>+<git sha 12>`; `GIT_SHA`/`RELEASE_TAG` come from Docker build args, and the deploy workflows register the same id in Sentry with `getsentry/action-release` (see `docs/runbooks/sentry-release-tracking.md`).
- Sampling/profiling/PII: `SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_PROFILE_SESSION_SAMPLE_RATE`, `SENTRY_PROFILE_LIFECYCLE`, `SENTRY_SEND_DEFAULT_PII`, `SENTRY_ERROR_SAMPLE_RATE` (config.py; defaults preserve current behavior).
- Trace propagation: `CeleryIntegration(propagate_traces=True)` links API to worker as one trace; poll-loop re-enqueues pass `SENTRY_NO_PROPAGATE_HEADERS` (`celery/tasks/job_execution.py`) to start a fresh trace.
- Error capture: generic handler in `core/exception_handlers.py` calls `capture_exception`.
- Tenant impact (both in `core/telemetry/context.py`, called from `api/deps.py`): `set_request_log_context` puts org/project in the log context + Sentry tags; `bind_sentry_user` binds user/org/project via `sentry_sdk.set_user` so issues report users/orgs affected. The user binding is scope-only — ids in the log context would stamp per-user cardinality on every INFO record `enable_logs` ships. Per-request identity is the `correlation_id` tag (`core/middleware.py`), not the user binding.
- Crons: `@sentry_sdk.monitor` on `/cron/*` endpoints (`api/routes/cron.py`).
- Runbook (alerts/dashboards/debugging): `features/sentry-utilization/SENTRY-RUNBOOK.md`.

## Telemetry / misc
- `core/telemetry/` package, public API re-exported from `__init__.py`: `tracing.py` (OTel tracer provider, span noise filter, FastAPI instrumentation with `exclude_spans` for ASGI send/receive, `OTEL_SEMCONV_STABILITY_OPT_IN=http/dup` default so server spans carry both old and stable `http.*` names, root-span attributes forwarded into Sentry transaction data, root server spans without `http.route` dropped as scanner/bot traffic, flush), `context.py` (log context, tenant tags, user binding), `metrics.py` (Sentry metric emit, stale-job and rate-threshold monitors), `http.py` (HTTP request metrics, called from `core/middleware.py`; unrouted requests emit only `http.server.request.unmatched` with no route label and no access log), `llm.py` (gen_ai span attributes, LLM call metrics), `db.py` (SQLAlchemy engine hooks, DB spans `db.statement` / `db.rows_affected`, pool/slow-query/transaction metrics).
- OTel auto-instrumentation (in `setup_telemetry`): FastAPI, SQLAlchemy, httpx, requests, logging, Celery (Queues insight), Redis (Caches insight), botocore (S3/KMS spans).
- `core/rate_monitor.py` — provider rate tracking
