# Service conventions (`app/services/`)

Authoritative conventions for business-logic services in kaapi-backend. Services live in
`app/services/<domain>/` (auth, collections, doctransform, llm, evaluations, response, ...).
Services are where orchestration happens — they call CRUD for DB work and call external HTTP
libraries for third-party APIs.

## What goes here (and what doesn't)

| Belongs in `services/`                                                   | Belongs elsewhere                        |
| ------------------------------------------------------------------------ | ---------------------------------------- |
| `httpx` / `openai` / `boto3` calls                                       | DB queries → `crud/`                     |
| Multi-step workflows (ingest a doc, then enqueue embedding, then notify) | Raw FastAPI deps → routes                |
| Domain validation that spans multiple records                            | Single-field validation → Pydantic model |
| Cost / token accounting, retries with backoff                            | Long-running async work → Celery task    |
| Translating CRUD return values into domain results                       | Schema definitions → models              |

## Hard rules

- **External HTTP must validate URLs you fetch.** Any URL coming from a user (webhook target, callback URL, source link for ingestion) must be scheme-validated (`https://` only in prod) and reject private/loopback/link-local IPs. SSRF is a blocker, not a follow-up.
- **`try` wraps only the throwing line(s).** Big try blocks are the #1 source of swallowed 404s becoming 500s.
- **Concrete exception types** — `except httpx.HTTPStatusError as e:`, not `except Exception`.
- **Logger prefix:** every line is `logger.info(f"[function_name] Message | key: {value}")`. Mask credentials / API keys / hashes / emails with `mask_string` from `app.utils`. Log start + finish of external HTTP calls and any retry.
- **Keyword-only args** for anything more than `(session, x)`, matching the CRUD convention.
- **Type hints on every parameter and return.** No `-> Any`.

## `HTTPException` in services

It's acceptable here — `services/auth.py` raises `HTTPException` directly when the domain failure maps cleanly to an HTTP status. Use it sparingly; when the same service may be called from a Celery task or CLI, prefer a domain exception that the route layer translates.

## Canonical example (from `app/services/auth.py`)

```python
import logging
from datetime import timedelta

from app.core import security
from app.core.config import settings
from app.utils import mask_string

logger = logging.getLogger(__name__)


def create_token_pair(
    user_id: int,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> tuple[str, str]:
    """Create an access token and refresh token pair."""
    access_token = security.create_access_token(
        user_id,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        organization_id=organization_id,
        project_id=project_id,
    )
    refresh_token = security.create_refresh_token(
        user_id,
        expires_delta=timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES),
        organization_id=organization_id,
        project_id=project_id,
    )
    logger.info(f"[create_token_pair] Token pair issued | user_id: {user_id}, access_token: {mask_string(access_token)}")
    return access_token, refresh_token
```

Delegation to `app.core.security` is the pattern — services orchestrate; primitives live in `app/core/`.

## External HTTP — checklist

When wrapping an external SDK or raw HTTP call, follow `.claude/conventions/error-handling.md` for the full pattern (source-tagged `[KAAPI]`/`[<PROVIDER>]` messages, fault-based log levels, typed exception ladders).

- **Timeout** — every `httpx`/`requests` call has an explicit timeout. The default is too long.
- **Retry policy** — idempotent GETs can retry with backoff. Mutations should retry only if you're certain the API is idempotent or you have an idempotency key.
- **Error mapping** — `httpx.HTTPStatusError` → a domain exception or `HTTPException` with a sensible code (often 502 for upstream failures, NOT 500).
- **Mock at this boundary in tests** — `monkeypatch` the HTTP client, not the DB. (See `test-writer` agent.)

## Calling CRUD

- Services own the `session` lifecycle for the operation. Call CRUD functions with `session=session` keyword arg.
- If CRUD returns `None`, decide whether that's a domain error (`raise NotFoundError`), a 404 (`raise HTTPException(404)`), or a silent skip (`return early`). Be explicit.

## Config / secrets

- Read from `settings` (`app.core.config`). Never read `os.environ` directly in a service.
- Defaults should lean toward cheap/safe: smallest model, lowest token cap, shortest TTL. Aggressive defaults belong in env, not code.

## Reuse existing helpers

The grep-first rule applies to _functions_, not just literals. Before hand-writing a helper that does anything generic — instantiating an external SDK client, mapping that SDK's exceptions, building or parsing a domain payload, reading/writing cloud storage, loading config — grep the whole tree first. These almost always exist already, and rarely in a neighbor file, so search by behavior (the SDK class, the model type, the storage call), not by directory.

Signs you're about to reinvent: instantiating a vendor client inline (`Anthropic(...)`, `OpenAI(...)`, `boto3.client(...)`); a long `except`-ladder over an SDK's error types; chains of `.get(...).get(...)` over a payload that has a typed model elsewhere. Prefer calling the existing function over reimplementing it; reach for a new helper only once you've confirmed none fits.

## Magic values

If you write the string `"openai"` or `"text-embedding-3-small"` or `1_000_000` in a service, ask whether it should be a constant / Enum / setting. Grep for the same literal — if it appears elsewhere, it should already be a constant.

## What you DO NOT do

- Don't write SQL directly — call CRUD.
- Don't import `fastapi.APIRouter` or define routes.
- Don't write long-running blocking loops — that's a Celery task.
- Don't call `time.sleep` inside an `async def` (use `asyncio.sleep`).
- Don't catch `HTTPException` from a sub-call and swallow it — propagate.
