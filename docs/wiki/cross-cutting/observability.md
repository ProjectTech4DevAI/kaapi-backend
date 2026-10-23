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
- Sampling/PII: `SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_SEND_DEFAULT_PII`, `SENTRY_ERROR_SAMPLE_RATE` (config.py; defaults preserve current behavior).
- Trace propagation: OTel's `CeleryInstrumentor` parents the worker's `run/...` span to `apply_async`; never inject/extract trace context by hand around tasks (it reparents task spans to the API request and Sentry splits them into separate transactions). `CeleryIntegration(propagate_traces=True)` links API to worker as one trace; poll-loop re-enqueues pass `SENTRY_NO_PROPAGATE_HEADERS` (`celery/tasks/job_execution.py`) to start a fresh trace.
- Error capture: handlers in `core/exception_handlers.py` report via `logger.error(exc_info=exc)` (5xx only). Don't add `capture_exception` next to it: on sentry-sdk 2.54 Dedupe does not merge the two, so each error becomes two events.
- Tenant impact (both in `core/telemetry/context.py`, called from `api/deps.py`): `set_request_log_context` puts org/project in the log context + Sentry tags; `bind_sentry_user` binds user/org/project via `sentry_sdk.set_user` so issues report users/orgs affected. The user binding is scope-only — ids in the log context would stamp per-user cardinality on every INFO record `enable_logs` ships. Per-request identity is the `correlation_id` tag (`core/middleware.py`), not the user binding.
- Crons: `@sentry_sdk.monitor` on `/cron/*` endpoints (`api/routes/cron.py`), e.g. `llm-call-retention-cron-job` (daily 09:00 IST).
- Runbook (alerts/dashboards/debugging): `features/sentry-utilization/SENTRY-RUNBOOK.md`.

## Telemetry / misc
- `core/telemetry/` package, public API re-exported from `__init__.py`: `tracing.py` (OTel tracer provider, span noise filter, FastAPI instrumentation with `exclude_spans` for ASGI send/receive, `OTEL_SEMCONV_STABILITY_OPT_IN=http/dup` default so server spans carry both old and stable `http.*` names, root-span attributes forwarded into Sentry transaction data, root server spans without `http.route` dropped as scanner/bot traffic, flush), `context.py` (log context, tenant tags, user binding), `metrics.py` (Sentry metric emit, stale-job and rate-threshold monitors), `http.py` (HTTP request metrics, called from `core/middleware.py`; unrouted requests emit only `http.server.request.unmatched` with no route label and no access log), `llm.py` (gen_ai span attributes, LLM call metrics), `db.py` (SQLAlchemy engine hooks, DB spans `db.statement` / `db.rows_affected`, pool/slow-query/transaction metrics).
- Log records carry `module_name` = `[<module>] <logger name>.<function>` (modules: knowledgebase, llm-call, llm-chain, config-version, assessment (API path), evals (2.0), fastevals). Module names are `LogModuleEnum`; `core/telemetry/context.py::_LOGGER_MODULES` maps each logging file's exact logger name (`__name__`) to its module, so a new logging file must be added there. A flow that runs another flow's code sets `log_context(module=LogModuleEnum.X)` at its entry (LLM chain jobs, fast-eval Celery tasks). Unmapped loggers (legacy assessment, shared eval helpers outside a fast-eval task) get `<logger name>.<function>` without the `[<module>]` prefix.
- Example, one log line in both sinks:
  - Sentry Logs: `module_name: [knowledgebase] app.api.routes.collection_job.get_collection_job`, `logger.name: app.api.routes.collection_job`, `code.line.number: 58`; filter with `module_name:[knowledgebase]*`.
  - CloudWatch (stdout format puts `module_name` where the logger name was): `2026-10-11 10:16:57,137 - [kaapi-backend] - [a1b2c3] - INFO - [knowledgebase] app.api.routes.collection_job.get_collection_job - [get_collection_job] Fetched job | job_id=8f2c`; Logs Insights `filter @message like "[knowledgebase]"`.
- `OTEL_ENABLED` is checked only at the entry points (`main.py`, `celery/celery_app.py`, `core/db.py`), which decide whether to call `setup_telemetry` / `instrument_app` / `instrument_db_engine`; the telemetry functions themselves are unconditional. `LogContextFilter` is installed by `configure_logging`, so log context works with OTel off.
- OTel auto-instrumentation (in `setup_telemetry`): FastAPI, SQLAlchemy, httpx, requests, logging, Celery (Queues insight), botocore (S3/KMS spans). Redis is deliberately not instrumented (only Celery result bookkeeping and rate-limit counters); keep Celery's `apply_async` span, it links API and worker traces.
- `core/rate_monitor.py` — provider rate tracking
