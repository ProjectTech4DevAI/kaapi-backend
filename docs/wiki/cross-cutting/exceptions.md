# Cross-cutting: Exceptions & Error Handling

All paths relative to `backend/app/`.

## Global handlers
- `core/exception_handlers.py` — registered on the FastAPI app; maps exception types to HTTP responses.
- `core/middleware.py` — request-level middleware.

## Provider/SDK error convention
- `.claude/conventions/error-handling.md` — the standardized pattern for wrapping provider/SDK exceptions at service/crud call sites. Follow it for any new OpenAI/Gemini/Anthropic call site.

## Outbound retry policy
- `core/retry.py`: `is_transient(error)` (5xx or a connection that never reached upstream). Call sites use tenacity `@retry(stop=stop_after_attempt(4), retry=retry_if_exception(is_transient))` directly: `utils.send_callback`, `services/llm/jobs.py` and `services/llm/guardrails.py`. Batch providers keep their own policy.

## Route-level rules
- Routes return structured error codes + human-readable messages (see any module page's endpoint errors).
- Per-row/batch work isolates failures: one item's provider error must not fail sibling items (pattern used across evaluations and batch processing).
