# Model conventions (`app/models/`)

Authoritative conventions for SQLModel entities in kaapi-backend. Models live in `app/models/`
and follow a strict house style. Read `app/models/user.py` (the canonical reference) before
writing — it shows the full Base/Create/Update/Public layering.

## Required structure for a new entity `Foo`

```python
class FooBase(SQLModel):
    """Shared fields between create, update, public, and table."""
    name: str = Field(
        max_length=255,
        sa_column_kwargs={"comment": "Human-readable name shown in the UI"},
    )
    status: FooStatusEnum = Field(
        sa_column_kwargs={"comment": "Lifecycle state: pending, active, archived"},
    )


class FooCreate(FooBase):
    """Payload accepted on POST."""
    # only fields the client must / may supply on create


class FooUpdate(SQLModel):
    """Payload accepted on PATCH — every field optional."""
    name: str | None = Field(default=None, max_length=255)
    status: FooStatusEnum | None = None


class Foo(FooBase, table=True):
    """DB row."""
    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier"},
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Tenant org that owns this foo"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the foo was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the foo was last updated", "onupdate": now},
    )


class FooPublic(FooBase):
    """Shape returned by the API."""
    id: int
    inserted_at: datetime
    updated_at: datetime


class FoosPublic(SQLModel):
    data: list[FooPublic]
    count: int
```

## Hard rules

- **Every `Field(...)` gets `sa_column_kwargs={"comment": "..."}`.** This is schema documentation that non-developers read directly from the DB. Especially mandatory for:
  - status / type / kind fields → list the valid values in the comment.
  - JSON columns → describe the expected structure.
  - Foreign keys → name the relationship.
  - Anything whose purpose isn't obvious from the name.
  - The comment must **add information**, not echo the field — `comment="The user's email address"` on an `email` field is noise. State the non-obvious: valid values, units, format, the relationship, or a constraint. If the only honest comment would restate the name, the field is self-documenting and a terse comment is fine — don't pad it.
- **Timestamps are `inserted_at` and `updated_at`.** NOT `created_at`. Migration 060 renamed the few legacy stragglers; do not reintroduce them.
- **Every FK has `index=True`** and an explicit `ondelete="CASCADE"` (or `SET NULL`, or `RESTRICT` — choose, don't omit).
- **Enums end in `Enum`.** `FooStatusEnum`, not `FooStatus`. snake_case for values when stored as strings.
- **JSON columns are for opaque metadata only.** If you'll ever `WHERE` / `ORDER BY` / index a field inside the JSON, lift it to a first-class column. Tell the user when you make this call.
- **Type hints use `|` unions** (Python 3.10+): `str | None`, not `Optional[str]`.

## Naming

- Class name = singular PascalCase: `Foo`, `Document`, `ApiKey`.
- Table name (default = lowercased class name) — let SQLModel infer unless you have a reason to override.
- Plural Public wrapper: `FoosPublic` (matches `UsersPublic`).
- Enum values: lowercase snake_case strings if string-valued.

## Validation

- Validate at the model layer with `Field(min_length=..., max_length=..., regex=..., ge=..., le=...)`. Don't push trivial validation into routes.
- `EmailStr` from `pydantic` for emails, not `str`.
- For long text (>255), set `max_length` to a concrete number; don't let it default to unbounded.

## Indexes

- `index=True` on any column you will filter, sort, or join on — every FK, every "lookup by X" column.
- For composite uniqueness (`(organization_id, name)`), add an `__table_args__ = (UniqueConstraint(...),)` block. Don't rely on app-level checks.

## What you DO NOT do

- Don't write the migration *in the model file* — the model is pure data shape. The matching migration is a separate file under `app/alembic/versions/`; write it (per `.claude/conventions/migration.md`) right after the model, using rev-id `NNN+1`.
- Don't import from `fastapi`, `app.crud`, or `app.services` in a model file. Models are leaf nodes.
- Don't use `setattr` on instances of these models. Use `model_copy(update={...})` or `sqlmodel_update(...)` (see `app/crud/user.py:update_user` for the pattern).
- Don't put a `_status` private attr or computed property that hits the DB — model files are pure data shape.
- Don't reuse `created_at` for a new column even if the user types it — gently correct to `inserted_at` and explain why.
