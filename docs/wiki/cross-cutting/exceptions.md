# Cross-cutting: Exceptions & Error Handling

All paths relative to `backend/app/`.

## Global handlers
- `core/exception_handlers.py` — registered on the FastAPI app; maps exception types to HTTP responses. The `Exception` catch-all returns a fixed message, never `str(exc)`, so provider bodies and DB text don't reach callers. Unhandled `SQLAlchemyError` gets its own handler: a "Database error: ..." hint with 409 (unique violation, SQLSTATE 23505), 422 (foreign key 23503, or `DataError`), 503 (`OperationalError`) or 500 (any other); 4xx logs at warning, 5xx at error; SQL and parameters only go to logs.
- DB errors are classified only in `db_error_handler`: crud/services let `SQLAlchemyError` propagate (rollback, then `raise`), never catch `IntegrityError` to build a 409 or match error text. A unique violation on a constraint listed in `_UNIQUE_CONSTRAINT_DETAILS` gets that message; add the constraint there for a specific 409. Sentry events carry `http.status_code`, `error.class` and `db.constraint` tags. Exception: eval `run_name` keeps its own 409 (`run_name_already_exists` is a client contract).
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
