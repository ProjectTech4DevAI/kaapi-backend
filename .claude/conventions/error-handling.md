# Error-handling & humane-logging conventions

Kaapi's standardized error-handling pattern. Apply when adding or refactoring error handling in:
- LLM/AI provider wrappers (`app/services/llm/providers/*.py`)
- CRUD layers that call external SDKs (`app/crud/**/*.py`)
- Any code that calls an external SDK or makes raw HTTP requests

## Core principles

1. **Every error path logs AND bubbles.** Never silently `return None, "..."` or silently `raise`. Log first (with `exc_info=True` where applicable), then return/raise.
2. **Tag every error message by source.**
   - `[KAAPI]` — failures originating in our backend: input validation, missing config, response-shape checks, post-processing failures, network-side timeouts/connection errors (we couldn't reach the provider), unexpected non-SDK errors.
   - `[<PROVIDER>]` — failures the provider returned to us (`[OPENAI]`, `[GEMINI]`, `[ANTHROPIC]`, `[SARVAM]`, `[ELEVENLABS]`, `[VERTEX]`). HTTP 4xx/5xx responses, malformed payloads, server overload.
3. **Messages are descriptive prose, not error codes alone.** State what failed, the most likely cause, and what the caller should do (retry, fix config, contact Kaapi).
4. **Always include `(code: …)` somewhere.** HTTP status (`code: 400`), exception class name for network errors (`code: ReadTimeout`), or provider status string (`code: 429 RESOURCE_EXHAUSTED`).
5. **Single source of truth for SDK errors.** Typed exception ladders live in the outermost dispatch (`execute()`), not duplicated inside each `_execute_<type>` method. Inner methods only handle Kaapi-side validation and response-shape checks; SDK exceptions bubble up.

## Provider taxonomy — pick the matching shape

| SDK shape | Examples | How to dispatch |
|---|---|---|
| **Typed-per-status exception classes** | OpenAI, Anthropic, Sarvam, ElevenLabs, Claude | One `except <Provider>.<NameError>` per status code. Order matters: subclasses before parents (e.g. `APITimeoutError` before `APIConnectionError`). |
| **Status-code dispatch on a single class** | Gemini (`ClientError`/`ServerError`) | Catch the umbrella class, branch on `.code` inside. |
| **Raw HTTP / `requests`** | Vertex AI (no SDK) | Map error logic inside the `_post()` helper. Handle `requests.Timeout` / `ConnectionError` / `RequestException` as `[KAAPI]`. Parse the provider's standard error envelope (e.g. Google's `{"error": {"code","status","message"}}`). Branch on `status_code`. |
| **CRUD that raises instead of returning** | `OpenAIVectorStoreCrud.update` raising `InterruptedError` | Same pattern, but `log + raise` instead of `log + return tuple`. |

## Status code → user message guidance

Use these as templates — adjust the verb and hint to the provider's specific terminology:

| Code | Tag | Wording template |
|---|---|---|
| 400 | `[<PROV>]` Bad request | "Review your config parameters and input payload — the request shape, model, or content may be invalid." |
| 401 | `[<PROV>]` Authentication failed | "Verify the API key is valid, has not expired, and has been correctly configured for this project." |
| 403 | `[<PROV>]` Permission denied | "The API key does not have access to the requested model or feature — check your plan and key scopes." |
| 404 | `[<PROV>]` Resource not found | "Verify the model name and any referenced IDs in your config are correct and available on your plan." |
| 409 | `[<PROV>]` Conflict | "The request conflicts with the current resource state — review concurrent requests before retrying." |
| 413 | `[<PROV>]` Request too large | "The input payload exceeds the provider's size limit — reduce prompt length, shrink attached files, or upload via Files API." |
| 422 | `[<PROV>]` Unprocessable entity | "Provider rejected the request payload — check input format and parameter values against the API spec." |
| 425 | `[<PROV>]` Too early | "Provider is not yet ready to process this request — wait a few seconds and retry." |
| 429 | `[<PROV>]` Rate limit / quota exceeded | "You have hit the provider's request rate or quota — wait at least 1 minute and retry. Request a quota increase or contact Kaapi if persistent." |
| 500 | `[<PROV>]` Server error | "Typically transient — retry in a few seconds. If issue persists, contact Kaapi." |
| 503 | `[<PROV>]` Service unavailable | "Provider is temporarily down or overloaded — retry in a few seconds." |
| 504 | `[<PROV>]` Deadline exceeded | "Provider took too long to complete the request — retry with a smaller payload." |
| 529 | `[<PROV>]` Overloaded (Anthropic) | "Anthropic infrastructure currently overloaded — retry with exponential backoff." |
| Network timeout | `[KAAPI]` `(code: ReadTimeout / ConnectTimeout / APITimeoutError)` | "Request timed out — retry with a smaller payload. If persistent, contact Kaapi." |
| Network conn | `[KAAPI]` `(code: ConnectionError / APIConnectionError)` | "Network/DNS issue reaching provider — check connectivity. If persistent, contact Kaapi." |

## Logging conventions

### Level — fault-based split

Always pick the level by **who is at fault**, not by "did the operation fail." Failure alone doesn't justify `.error`; alerting depends on it.

| Failure type | Level | Why |
|---|---|---|
| 4xx (400/401/403/404/409/413/422/425/429) | `warning` | Caller's fault — bad input, expired key, hit quota. Common at scale; alerting on every 429 would be noise. |
| 5xx (500/502/503/504/529) | **`error`** | Provider broke. A spike *is* worth knowing about — Sentry/alerting should catch this. |
| Network (`Timeout`, `ConnectionError`, `RequestException`, SDK `APITimeoutError`/`APIConnectionError`) | **`error`** | We couldn't reach them — infra/networking issue, escalation-worthy. |
| Response-shape (missing `response_id`/text/audio, schema validation) | `warning` | Often safety-filter / quota / bad input issue, not a bug. |
| Kaapi-side validation (wrong type, missing field, unsupported value) | `warning` | Caller's fault — same logic as 4xx. |
| Post-processing failure (audio conversion, GCS upload, base64 decode of provider output) | **`error`** | Our code or infra broke. |
| Generic `Exception` catch-all | **`error`** | By definition unexpected; treat as a bug until proven otherwise. |
| Provider catch-all of unknown status (e.g. `APIStatusError`, `ApiError`) | **branched** — see below | Status code is known at catch time; pick level accordingly. |

### Branched level in catch-alls

When a single `except` block covers both 4xx and 5xx (e.g. `APIStatusError`, `ApiError`, the non-OK branch of an HTTP `_post()`), pick the level from `status_code`:

```python
log = logger.error if status and status >= 500 else logger.warning
log(
    f"[<ClassName>.<method>] {error_message} | provider={provider}, ...",
    exc_info=True,
)
```

Put a one-line comment above the assignment explaining why, so future readers don't "simplify" it back to a single level.

### Other conventions

```python
logger.warning(  # or logger.error per the table above
    f"[<ClassName>.<method_name>] {error_message} | "
    f"provider={provider}, type={completion_type}, model={model}",
    exc_info=True,  # include when an exception was caught (not for pure validation failures)
)
```

Always include in the structured tail: `provider=`, the call type/method, `model=` (where known), `request_id=` or response_id when available. These are the join keys ops needs.

**`exc_info=True`** belongs on any path that caught a real exception. Skip it on Kaapi-side validation that just builds a string and returns (no exception was raised).

## Required structure (provider with `execute()` dispatch)

```python
def execute(self, ...):
    provider = completion_config.provider
    completion_type = completion_config.type
    try:
        if completion_type == "stt":   return self._execute_stt(...)
        if completion_type == "tts":   return self._execute_tts(...)
        if completion_type == "text":  return self._execute_text(...)
        # Unsupported type → [KAAPI] log + return
        ...

    # === SDK exceptions, typed ladder (one per status code) ===
    except <Provider>.BadRequestError as e:        # 400
    except <Provider>.AuthenticationError as e:    # 401
    except <Provider>.PermissionDeniedError as e:  # 403
    except <Provider>.NotFoundError as e:          # 404
    except <Provider>.ConflictError as e:          # 409
    except <Provider>.UnprocessableEntityError as e:  # 422
    except <Provider>.RateLimitError as e:         # 429
    except <Provider>.InternalServerError as e:    # 500
    # Network — subclass first
    except <Provider>.APITimeoutError as e:        # [KAAPI] + (code: type(e).__name__)
    except <Provider>.APIConnectionError as e:     # [KAAPI] + (code: type(e).__name__)
    except <Provider>.APIResponseValidationError as e:  # response schema
    except <Provider>.APIStatusError as e:         # catch-all 4xx/5xx; can branch by e.status_code for 413/503/504/529 etc.
    except <Provider>.AnthropicError as e:         # SDK base catch-all

    # === Kaapi-side ===
    except TypeError as e:    # [KAAPI] config signature mismatch
    except ValueError as e:   # [KAAPI] validation (e.g. _parse_input)
    except Exception as e:    # [KAAPI] unexpected
```

Inner `_execute_*` methods handle ONLY:
- Input-type validation (`[KAAPI]`)
- Missing required-field checks (`[KAAPI]`)
- Response-shape checks like missing `response_id`, missing text, missing audio (`[<PROVIDER>]`)
- Audio/format post-processing failures (`[KAAPI]`)

They do NOT wrap SDK calls in `try/except` — exceptions bubble up to `execute()`.

## Required structure (raw HTTP via `requests`)

Centralize HTTP error mapping inside `_post()` (or equivalent). The caller just does:

```python
data, err = self._post(model, payload, log_context=f"provider={provider}, type=stt")
if err:
    return None, err  # err is already-logged and tagged
```

Inside `_post()`:

```python
try:
    resp = requests.post(url, ..., timeout=REQUEST_TIMEOUT)
except requests.Timeout as e:       # [KAAPI] (code: ReadTimeout / ConnectTimeout)
except requests.ConnectionError as e:  # [KAAPI] (code: ConnectionError)
except requests.RequestException as e: # [KAAPI] (code: <subclass>)

if not resp.ok:
    # Parse provider error envelope, e.g.:
    #   Google: resp.json()["error"] -> {"code","status","message"}
    # Branch on resp.status_code → 400/401-403/404/429/5xx/other
    # Build [<PROVIDER>] message including code AND provider's status string
```

## Required structure (CRUD that `raise`s)

Same pattern, but build `error_message` first, log it, then raise carrying the same string:

```python
except openai.RateLimitError as e:
    error_message = f"[OPENAI] Rate limit exceeded (code: {e.status_code}): {e.message}. ..."
    logger.warning(
        f"[<ClassName>.<method>] {error_message} | <context tail>",
        exc_info=True,
    )
    raise InterruptedError(error_message)
```

## Surface request_id when available

Whenever the SDK exposes a per-request identifier (OpenAI/Anthropic's `request_id`, Sarvam's `request_id`, ElevenLabs' `transcription_id`, Vertex's `responseId`), include it inside the error message or the log tail. Support escalation paths depend on it.

## Don'ts

- ❌ Don't return a bare error from validation without logging it.
- ❌ Don't use HTTP-adjacent status codes for non-HTTP failures (e.g. don't tag a `ConnectTimeout` as `(code: 408)`). Use the exception class name.
- ❌ Don't duplicate typed exception ladders inside each `_execute_*` and also in `execute()`. Pick one location (almost always `execute()`).
- ❌ Don't bare-message the catch-alls: `"Unexpected error occurred"` is useless. Include `str(e)`, the operation, and a "contact Kaapi" hint.
- ❌ Don't catch `APITimeoutError` AFTER `APIConnectionError` — the parent shadows the child.
- ❌ Don't import provider exception classes from underscore-prefixed private modules (e.g. `anthropic._exceptions`). If a class isn't re-exported, branch on `status_code` inside the parent catch-all instead.
- ❌ Don't log unconditionally for both success and failure paths (a common bug — verify the `logger.error(...)` line sits inside the `if err:` block, not above it).
- ❌ Don't log every failure as `.warning` "because the operation failed." Pick level by **fault** (see Logging conventions table). Ops alerts fire on `.error` rate; everything-as-warning means real outages get buried.
- ❌ Don't log every failure as `.error` either. `.error` on every 429 / 401 / 400 generates pager noise — those are the caller's fault.

## Workflow when applying this pattern

1. **Identify the provider's exception shape** — read the SDK's error module (e.g. `<provider>/errors/__init__.py` or `<provider>/_exceptions.py`) to enumerate available classes and what status codes they map to.
2. **Identify the Kaapi-side error sites** — every existing `return None, "<string>"` and every `raise <Error>(...)` without a preceding `logger.*` call.
3. **Check for doubling** — if typed handlers exist in inner methods, move them up to `execute()`/equivalent and let exceptions bubble.
4. **Apply the matrix above** — pick a row per error site, write a descriptive message with tag + code + cause + remediation, add `logger.warning(..., exc_info=True)` (or `.error` for unexpected), then return/raise.
5. **Surface `request_id` / response_id** in messages or log tails wherever the SDK exposes one.
6. **Sanity check ordering** — `APITimeoutError` before `APIConnectionError`; specific subclasses before `<Provider>ApiError` / `APIStatusError` catch-alls.
7. **Run a syntax check**: `python -c "import ast; ast.parse(open('<file>').read()); print('OK')"`.

## Reference implementations in this repo

When in doubt, mirror the file that matches the SDK shape:

- **Typed-per-status SDK** → `app/services/llm/providers/oai.py`, `claude.py`, `sai.py`, `eai.py`
- **Status-code dispatch on umbrella class** → `app/services/llm/providers/gai.py`
- **Raw HTTP** → `app/services/llm/providers/gai_vertex.py`
- **CRUD that raises** → `app/crud/rag/open_ai.py` (`OpenAIVectorStoreCrud.update`)
