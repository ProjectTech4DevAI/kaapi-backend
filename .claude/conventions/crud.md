# CRUD conventions (`app/crud/`)

Authoritative conventions for data-access functions in kaapi-backend. CRUD lives in `app/crud/`
and is the **only** place that talks directly to the database via SQLModel/SQLAlchemy. Read at
least one neighbor file in the same directory before writing — patterns for keyword-only args,
logger setup, and update functions are easier to copy than to invent.

## Hard rules

- **No `HTTPException` in this layer.** Ever. Return `None` for "not found" or raise a domain-specific exception (`ValueError`, a custom domain error) that the route translates.
- **No third-party HTTP calls.** No `httpx`, no `openai`, no boto3, no `requests`. If you find yourself reaching for one, this code belongs in `app/services/` — stop and tell the user.
- **No business logic.** Validation, orchestration, multi-step workflows → services. CRUD is "read this row, write this row, list these rows with filters".
- **No `print`. Use `logger`.** Module top: `import logging; logger = logging.getLogger(__name__)`. Every line is `logger.info(f"[function_name] Message | key: {value}")`. Mask sensitive values with `mask_string` from `app.utils` — e.g. `f"... | email: {mask_string(email)}"`.

## Canonical function shape (from `app/crud/user.py`)

```python
def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    logger.info(f"[create_user] User created | user_id: {db_obj.id}")
    return db_obj


def get_user_by_email(*, session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()
```

Note: **keyword-only args** with `*` for anything more than `(session, id)`. Reduces argument-order bugs at call sites.

## Naming

- `get_<thing>_by_<key>` returns one or `None`.
- `list_<things>(...)` returns a list (plural in the name matches plural in the return).
- `create_<thing>`, `update_<thing>`, `delete_<thing>`.
- `bulk_<verb>_<things>` for batch ops.
- No `_one` / `_all` suffixes — the name should already say it.

## Performance

- **N+1 is a bug.** If you `list_<things>` and the caller is going to access a relationship attribute, eager-load with `selectinload(...)` or `joinedload(...)`. Read the call sites before deciding.
- **Index any column you filter on.** That's a model-layer concern (`.claude/conventions/model.md`), but if you write a `get_<thing>_by_<column>` and the column has no index, flag it.
- **Pagination.** Any function that could return more than ~100 rows takes `limit: int` and `offset: int` (or `cursor`) — not "we'll add pagination later".

## Concurrency

- "Compute next / check then write" is a race condition. `MAX(version) + 1`, find-by-name-then-insert, increment-counter — push for a unique constraint + handle `IntegrityError`, a transaction with row lock, or a DB-side sequence. Tell the user before silently shipping the racy version.
- Don't `session.commit()` inside a loop. Build the list, add all, commit once.

## Error surface (what to raise, what to return)

For CRUD that wraps an external SDK (e.g. `OpenAIVectorStoreCrud`), follow `.claude/conventions/error-handling.md` — log the source-tagged message first, then raise carrying the same string.

| Situation | Return / raise |
|---|---|
| Not found | `return None` |
| Found multiple but exactly one was expected | `raise ValueError(...)` (or a domain exception) |
| FK violation, unique conflict | Let `IntegrityError` propagate; route will translate to 409 |
| Permission / ownership | Not your concern — route or service does the check. CRUD trusts its inputs. |

## SQL injection / shell injection

- Always use parameterized queries (SQLModel/SQLAlchemy does this for you with `where(...)`). **Never** f-string a value into raw SQL.
- If you must use `op.execute` or `text(...)`, use bound parameters.

## What you DO NOT do

- Don't import from `fastapi` (no `HTTPException`, no `Depends`).
- Don't import from `httpx`, `requests`, `openai`, cloud SDKs.
- Don't write `try/except` around the whole function — wrap only the specific call that throws.
- Don't catch `Exception` — use the concrete exception type.
