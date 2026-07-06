# Cross-cutting: Observability

All paths relative to `backend/app/`.

## Langfuse
- `core/langfuse/langfuse.py` — client integration. LLM calls and evaluation runs write traces; evaluation scores attach to traces (`scores` list, reasoning in score `comment`).
- Durable score maps on `evaluation_run` are the resync source of truth when a Langfuse write fails.

## Logging
- `core/logger.py`. Convention (CLAUDE.md): every log line starts with the function name in brackets — `logger.info(f"[function_name] Message | key: {value}")`.

## Sentry
- `core/sentry_filters.py` — event filtering before send.

## Telemetry / misc
- `core/telemetry.py`
- `core/rate_monitor.py` — provider rate tracking
