# Cross-cutting: Exceptions & Error Handling

All paths relative to `backend/app/`.

## Global handlers
- `core/exception_handlers.py` — registered on the FastAPI app; maps exception types to HTTP responses. The `Exception` catch-all returns a fixed message, never `str(exc)`, so provider bodies and DB text don't reach callers.
- `core/middleware.py` — request-level middleware.

## Status-code selection
Raise `HTTPException` directly with the right status code from routes, crud, and service. Never raise a bare `ValueError` for a caller-facing failure: nothing catches it, so it becomes a 500 with a generic body.

| Condition | Code |
|---|---|
| Resource absent (ours, or one we reference upstream) | 404 |
| State conflict (already exists, already deleted) | 409 |
| Payload unparseable or wrong shape | 422 |
| Valid shape, unacceptable value | 400 |
| Upstream errored or unreachable (Langfuse, OpenAI, Gemini, KMS, object storage) | 502 |
| Our async infra could not accept the work (Celery broker) | 503 |
| Our bug (DB write failed, unexpected exception) | 500 |

- A broad `except Exception` around a DB write must re-raise `HTTPException` first, otherwise a typed 4xx/5xx gets flattened to 500.

## Provider/SDK error convention
- `.claude/conventions/error-handling.md` — the standardized pattern for wrapping provider/SDK exceptions at service/crud call sites. Follow it for any new OpenAI/Gemini/Anthropic call site.

## Route-level rules
- Routes return structured error codes + human-readable messages (see any module page's endpoint errors).
- Per-row/batch work isolates failures: one item's provider error must not fail sibling items (pattern used across evaluations and batch processing).
