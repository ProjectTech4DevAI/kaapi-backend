# Route conventions (`app/api/routes/`)

Authoritative conventions for FastAPI endpoints in kaapi-backend. Routes live in
`app/api/routes/` and follow a strict house style. Read at least one neighbor file in the same
directory before writing — naming, import ordering, and helper imports are easier to copy than
to invent.

## Required ingredients for every endpoint

1. **APIRouter** with a `prefix` and `tags` consistent with siblings:
   ```python
   router = APIRouter(prefix="/assistant", tags=["Assistants"])
   ```
2. **`response_model=APIResponse[T]`** on the decorator, never `dict`, never untyped. Use the actual Pydantic / SQLModel return type, not `Any`.
3. **`status_code=201` / `204`** on create / delete; default 200 is fine for GET / PATCH.
4. **`description=load_description("<domain>/<action>.md")`** instead of inline docstrings for the swagger description. The matching markdown lives at `backend/app/api/docs/<domain>/<action>.md` — create it in the same change.
5. **`dependencies=[Depends(require_permission(Permission.XYZ))]`** when the endpoint is restricted. Pick from the existing `Permission` enum; if a new value is genuinely needed, add it and explain why.
6. **`SessionDep` and `AuthContextDep`** for db + current user/org/project. Never re-implement these.
7. **Return `APIResponse.success_response(...)`** at the end — never a raw model.
8. **Type hints on every parameter and the return.** Path/query params use `Annotated[..., Path(description=...)]` / `Annotated[..., Query(...)]`.

## Canonical example (matches `app/api/routes/users.py:120`)

```python
@router.get(
    "/me",
    description=load_description("users/get_me.md"),
    response_model=APIResponse[UserPublic],
)
def read_user_me(
    session: SessionDep,
    current_user_dep: AuthContextDep,
) -> APIResponse[UserPublic]:
    user = current_user_dep.user
    return APIResponse.success_response(user)
```

## Swagger markdown

For every new endpoint, create `backend/app/api/docs/<domain>/<action>.md`. Keep it terse — 1-3 short paragraphs. Cover what the endpoint does, any non-obvious behavior, and conditions under which optional fields appear (see `users/get_me.md` for the shape).

## Layering rules

- **Routes are thin.** Pull arguments, call a CRUD or service function, wrap the result. If your route has >20 lines of business logic, that logic belongs in `app/services/<domain>/`.
- **HTTPException is allowed here.** Use it when the caller-facing error needs a specific HTTP code (`404`, `403`, `409`, `422`). Catch domain exceptions from CRUD/services and translate.
- **Never call third-party HTTP from a route.** That belongs in `app/services/`.
- **Never write SQL or `session.exec(select(...))` in a route.** Use a CRUD function. If one doesn't exist, write it in the `crud` layer (per `.claude/conventions/crud.md`) before wiring the route.

## Status codes (the ones to get right)

- `201` on POST create.
- `204` on DELETE (no body — return nothing, not `APIResponse.success_response(None)`).
- `409` on conflict (unique constraint violation, duplicate name).
- `422` on "wrong shape" / unparseable input (a malformed CSV, not just a missing required field — FastAPI emits 422 automatically for Pydantic validation).
- `400` for genuinely "bad client input that's not a shape issue".
- Don't return 200 + `{"error": "..."}` — raise `HTTPException` with the right code.

## Ownership checks

Anywhere a route accepts an `id` that could refer to data outside the caller's scope, **verify ownership** before returning data:

```python
obj = get_thing_by_id(session, thing_id)
if obj is None or obj.organization_id != current_user.organization_.id:
    raise HTTPException(status_code=404, detail="Thing not found")
```

Returning `404` instead of `403` for cross-tenant access is intentional — it doesn't leak existence.

## Background work

- Short fire-and-forget (send an email, write an audit log) → `BackgroundTasks`.
- Heavy or retryable (LLM call, large doc transform, anything with timeouts) → Celery task in `app/celery/tasks/`, written per `.claude/conventions/celery.md`.

## Logging

`logger = logging.getLogger(__name__)` at the module top. Every line is `logger.info(f"[handler_name] Message | key: {value}")`. Log non-trivial actions (creates, deletes, ownership failures) — don't spam `info` on every GET. Mask sensitive values with `mask_string` from `app.utils`.

## What you DO NOT do

- Don't add the route registration in `app/api/main.py` (or wherever the aggregator lives) without checking the existing alphabetical / grouped order.
- Don't return raw `dict`, `JSONResponse`, or untyped responses.
- Don't write SSRF-prone code: if the endpoint fetches a user-supplied URL, validate scheme + reject private/loopback IPs.
- Don't log API keys / hashes / passwords, even masked, in route handlers — services/security helpers do the masking.
