# Error-handling conventions (external SDKs & raw HTTP)

Authoritative convention for exception handling in any kaapi-backend code that calls an external SDK
or makes raw HTTP requests — LLM provider wrappers (`app/services/llm/providers/*.py`), CRUD that
calls external SDKs (`app/crud/**`), or any raw `requests`/`httpx` call. This is a cross-cutting
concern of the **service** and **crud** layers; apply it whenever you write or refactor those call
sites.

**Mirror the closest reference file rather than reconstructing the structure from scratch** — they
are the source of truth for the full shape:

| SDK shape | Reference file |
|---|---|
| Typed-per-status exception classes (OpenAI, Anthropic, Sarvam, ElevenLabs) | `app/services/llm/providers/oai.py`, `claude.py`, `sai.py`, `eai.py` |
| Status-code dispatch on one umbrella class (Gemini) | `app/services/llm/providers/gai.py` |
| Raw HTTP / no SDK (Vertex) | `app/services/llm/providers/gai_vertex.py` |
| CRUD that `raise`s instead of returning | `app/crud/rag/open_ai.py` (`OpenAIVectorStoreCrud.update`) |

## Core principles

1. **Every error path logs AND bubbles.** Never silently `return None, "..."` or silently `raise`.
   Log first (with `exc_info=True` when an exception was caught), then return/raise.
2. **Tag every message by source.**
   - `[KAAPI]` — our fault: input validation, missing config, response-shape checks, post-processing
     failures, network timeouts/connection errors (we couldn't reach the provider), unexpected non-SDK errors.
   - `[<PROVIDER>]` — what the provider returned: HTTP 4xx/5xx, malformed payloads, overload.
     (`[OPENAI]`, `[GEMINI]`, `[ANTHROPIC]`, `[SARVAM]`, `[ELEVENLABS]`, `[VERTEX]`.)
3. **Descriptive prose, not bare codes.** State what failed, the likely cause, and what the caller
   should do (retry, fix config, contact Kaapi).
4. **Always include `(code: …)`** — HTTP status (`code: 400`), exception class for network errors
   (`code: ReadTimeout`), or provider status string (`code: 429 RESOURCE_EXHAUSTED`).
5. **Single source of truth for SDK errors.** The typed exception ladder lives in the outermost
   dispatch (`execute()`), not duplicated inside each `_execute_<type>`. Inner methods handle only
   Kaapi-side validation and response-shape checks; SDK exceptions bubble up to `execute()`.
6. **Surface `request_id` / response_id** in the message or log tail wherever the SDK exposes one —
   support escalation depends on it.

## Log level — by fault, not by "did it fail"

Pick the level by **who is at fault**. Failure alone doesn't justify `.error`; ops alerting fires on
`.error` rate, so everything-as-error buries real outages and everything-as-warning hides them.

| Failure | Level |
|---|---|
| 4xx (400/401/403/404/409/413/422/425/429) — caller's fault | `warning` |
| Kaapi-side validation / response-shape checks | `warning` |
| 5xx (500/502/503/504/529) — provider broke | **`error`** |
| Network (`Timeout`, `ConnectionError`, SDK `APITimeoutError`/`APIConnectionError`) | **`error`** |
| Post-processing failure (audio convert, GCS upload, base64 decode) | **`error`** |
| Generic `Exception` catch-all | **`error`** |

When one `except` covers both 4xx and 5xx (`APIStatusError`, `ApiError`, the non-OK branch of a raw
`_post()`), branch the level on `status_code` — and leave a one-line comment so nobody "simplifies"
it back to a single level:

```python
# 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
log = logger.error if status and status >= 500 else logger.warning
log(f"[<ClassName>.<method>] {error_message} | provider={provider}, ...", exc_info=True)
```

Log tail always carries the ops join keys: `provider=`, the call type/method, `model=` (where
known), and `request_id`/response_id when available. `exc_info=True` on any path that caught a real
exception; omit it on pure validation that just builds a string.

## Message templates (adjust verb/hint to the provider)

| Code | Tag | Wording |
|---|---|---|
| 400 | `[<PROV>]` | "Review your config and input payload — request shape, model, or content may be invalid." |
| 401 | `[<PROV>]` | "Verify the API key is valid, not expired, and configured for this project." |
| 403 | `[<PROV>]` | "The API key lacks access to the requested model/feature — check plan and key scopes." |
| 404 | `[<PROV>]` | "Verify the model name and any referenced IDs are correct and available on your plan." |
| 409 | `[<PROV>]` | "Request conflicts with current resource state — review concurrent requests before retrying." |
| 413 | `[<PROV>]` | "Payload exceeds the provider's size limit — reduce prompt, shrink files, or use the Files API." |
| 422 | `[<PROV>]` | "Provider rejected the payload — check input format and parameter values against the API spec." |
| 425 | `[<PROV>]` | "Provider not ready yet — wait a few seconds and retry." |
| 429 | `[<PROV>]` | "Hit the provider's rate/quota — wait ≥1 min and retry. Request a quota increase or contact Kaapi if persistent." |
| 500 | `[<PROV>]` | "Typically transient — retry in a few seconds. If it persists, contact Kaapi." |
| 503 | `[<PROV>]` | "Provider temporarily down or overloaded — retry in a few seconds." |
| 504 | `[<PROV>]` | "Provider took too long — retry with a smaller payload." |
| 529 | `[ANTHROPIC]` | "Anthropic infrastructure overloaded — retry with exponential backoff." |
| Network timeout | `[KAAPI]` `(code: ReadTimeout/ConnectTimeout/APITimeoutError)` | "Request timed out — retry smaller. If persistent, contact Kaapi." |
| Network conn | `[KAAPI]` `(code: ConnectionError/APIConnectionError)` | "Network/DNS issue reaching provider — check connectivity. If persistent, contact Kaapi." |

## Structure rules

- **Typed-ladder ordering:** specific subclasses before parents, and `APITimeoutError` **before**
  `APIConnectionError` (parent shadows child). Provider catch-alls (`APIStatusError`,
  `<Provider>Error`) last; branch by `status_code` inside them for 413/503/504/529.
- **Raw HTTP:** centralize mapping inside `_post()` — handle `Timeout`/`ConnectionError`/
  `RequestException` as `[KAAPI]`, parse the provider's error envelope, branch on `status_code`. The
  caller just does `data, err = self._post(...)` then `if err: return None, err`.
- **CRUD that raises:** build `error_message`, log it, then `raise <Error>(error_message)` carrying
  the same string.
- **Don't import provider exceptions from private modules** (`anthropic._exceptions`). If a class
  isn't re-exported, branch on `status_code` inside the parent catch-all.

## Don'ts

- ❌ Return a bare validation error without logging it.
- ❌ HTTP-style codes for non-HTTP failures (tag a `ConnectTimeout` as `(code: ReadTimeout)`, not `408`).
- ❌ Duplicate the typed ladder inside each `_execute_*` AND in `execute()`. Pick `execute()`.
- ❌ Bare catch-all messages (`"Unexpected error occurred"`). Include `str(e)`, the operation, a "contact Kaapi" hint.
- ❌ `.warning` everything "because it failed", or `.error` everything (pager noise on every 429/401). Level by fault.
- ❌ `logger.error` sitting above the `if err:` block so it fires on success too.

## Workflow

1. Read the SDK's error module to enumerate its exception classes and status mappings.
2. Find every Kaapi-side error site — each `return None, "<string>"` and each `raise` lacking a preceding `logger.*`.
3. If typed handlers exist in inner methods, move them up to `execute()` and let exceptions bubble.
4. Apply the tables: per site write tag + code + cause + remediation, then `logger.<level>(..., exc_info=...)`, then return/raise.
5. Sanity-check ordering (subclasses first; `APITimeoutError` before `APIConnectionError`).
6. Syntax check: `python -c "import ast; ast.parse(open('<file>').read()); print('OK')"`.
