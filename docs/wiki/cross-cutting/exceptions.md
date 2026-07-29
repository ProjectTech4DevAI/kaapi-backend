# Cross-cutting: Exceptions & Error Handling

All paths relative to `backend/app/`.

## Global handlers
- `core/exception_handlers.py` — registered on the FastAPI app; maps exception types to HTTP responses. The `Exception` catch-all returns a fixed message, never `str(exc)`, so provider bodies and DB text don't reach callers.
- `core/middleware.py` — request-level middleware.

## Domain exceptions
- `core/exceptions.py` — `KaapiError` base plus `NotFoundError` (404), `ConflictError` (409), `InvalidValueError` (400), `InvalidPayloadError` (422), `UpstreamError` (502, takes `provider=`), `ServiceUnavailableError` (503). Handled by `kaapi_error_handler`, which reads `status_code` off the class.
- Never raise a bare `ValueError` from crud/service for a caller-facing failure: nothing catches it, so it becomes a 500 with a generic body. Use one of the above.
- `KaapiError` is the preferred raise in crud/service (it keeps the layer transport-agnostic), but the migration is partial: ~78 `raise HTTPException` sites remain in `crud/` and are still correct as long as the code is right. Prefer `KaapiError` in new code; convert neighbours opportunistically rather than mixing both idioms inside one function.
- Code selection, applied across evaluations, knowledge-base, llm-call, tenancy, and platform:

| Condition | Code |
|---|---|
| Resource absent (ours, or one we reference upstream) | 404 |
| State conflict (already exists, already deleted) | 409 |
| Payload unparseable or wrong shape | 422 |
| Valid shape, unacceptable value | 400 |
| Upstream errored or unreachable (Langfuse, OpenAI, Gemini, KMS, object storage) | 502 |
| Our async infra could not accept the work (Celery broker) | 503 |
| Our bug (DB write failed, unexpected exception) | 500 |

- A broad `except Exception` around a DB write must re-raise `HTTPException`/`KaapiError` first, otherwise typed errors get flattened to 500.

## Provider/SDK error convention
- `.claude/conventions/error-handling.md` — the standardized pattern for wrapping provider/SDK exceptions at service/crud call sites. Follow it for any new OpenAI/Gemini/Anthropic call site.

## Route-level rules
- Routes return structured error codes + human-readable messages (see any module page's endpoint errors).
- Per-row/batch work isolates failures: one item's provider error must not fail sibling items (pattern used across evaluations and batch processing).
