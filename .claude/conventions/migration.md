# Migration conventions (`app/alembic/versions/`)

Authoritative conventions for Alembic migrations in kaapi-backend. The DB is PostgreSQL.
Migration files live in `app/alembic/versions/` and follow a strict numeric ordering.

## Before writing anything

1. `ls backend/app/alembic/versions/` and find the highest `NNN_*.py`. The new revision id is
   **that number + 1**, zero-padded to 3 digits. (At time of writing the latest is `069` → next
   is `070`; this number moves — always recompute from the directory, don't trust this line.) Do
   not skip numbers.
2. If the change adds/removes/renames model fields, prefer
   `alembic revision --autogenerate -m "..." --rev-id <NNN>` (run via `uv`, not `pip`) and then
   hand-edit. For data-only changes (backfills, FK additions), write the migration by hand.
3. Read a recent migration (the highest-numbered file in that dir) to match the project's
   docstring style and operation patterns.

## Required structure

```python
"""<one-line summary>

Revision ID: NNN
Revises: <previous>
Create Date: YYYY-MM-DD HH:MM:SS.000000

<Short paragraph(s) on the WHY: the reason for the change and any non-obvious
 ordering or gotcha a future reader debugging prod needs. Do NOT narrate the
 obvious operations — the reader can see `add_column` / `create_index` in the
 code; tell them what they can't see.>
"""

import sqlalchemy as sa
from alembic import op

revision = "NNN"
down_revision = "<previous>"
branch_labels = None
depends_on = None


def upgrade():
    ...


def downgrade():
    ...
```

## Hard rules

- **`downgrade()` is mandatory and must actually reverse `upgrade()`.** Empty `pass` is a blocker. If reverse is truly impossible (e.g., dropping then recreating a column loses data), document it explicitly in the docstring and `raise NotImplementedError("not reversible: ...")` in downgrade — but try harder first.
- **Backfills go inside `upgrade()` SQL** using `op.execute(...)`, not as a separate manual script. Same for cleanup of orphan rows before adding constraints.
- **New tables** must include:
  - `id` primary key.
  - `inserted_at` (NOT `created_at`) and `updated_at` timestamps. Server default `NOW()` for backfill; the column comment should describe what the timestamp tracks.
  - `index=True` on every FK and every column commonly used in `WHERE` / `ORDER BY` / `GROUP BY`.
  - `sa.Column(..., comment="...")` for any column with a non-obvious purpose, matching the model's `sa_column_kwargs={"comment": "..."}`.
- **Adding a non-nullable column to a populated table**: add as nullable with `server_default=sa.text("...")`, backfill, then `ALTER COLUMN ... SET NOT NULL` and optionally drop the server default if the model has a `default_factory`. See a recent migration for the exact pattern.
- **Adding a unique constraint to a populated table**: dedupe first (`op.execute("DELETE ... USING ...")`), then `CREATE UNIQUE INDEX ... CONCURRENTLY` and `ALTER TABLE ... ADD CONSTRAINT ... USING INDEX` so the build doesn't take `AccessExclusiveLock`.
- **Index builds on large tables**: use `CREATE INDEX CONCURRENTLY` via raw `op.execute(...)`. Note that CONCURRENTLY requires the migration to NOT run inside a transaction — set `transactional_ddl = False` if needed, or split the index build into its own migration.

## What to verify before declaring done

- `grep -n "down_revision" backend/app/alembic/versions/*.py` shows your `revision` is unique and the chain `... → <prev> → <yours>` is intact.
- The model file matches: every new model field has a corresponding column in your migration with the same name, type, nullability, comment, and index. Conversely every column you add exists in the model.
- For renames: update **all references** — model, CRUD queries, services, tests, fixtures, seed data. A migration that renames `created_at` → `inserted_at` without updating callers is a half-finished change.
- Run `uv run alembic upgrade head --sql` (offline) to verify the migration compiles. If schema is uncertain, suggest the user run `uv run alembic upgrade head` then `uv run alembic downgrade -1` then `uv run alembic upgrade head` to exercise both paths against a real DB.

## What you DO NOT do

- Don't add `HTTPException`, route handlers, business logic, or external HTTP calls in a migration.
- Don't write `print(...)` debug statements — use the migration docstring. If a long backfill genuinely needs progress logging, use `logging.getLogger("alembic.runtime.migration")` with the standard `[<revision_id>] ...` prefix.
- Don't skip the docstring. The docstring is what someone debugging at 2am will read.
- Don't pad the docstring either — keep it to the non-obvious WHY. A step-by-step recap of the `op.*` calls (which the reader can already see in `upgrade()`) is noise. Same for inline comments: explain a tricky backfill or lock-avoidance trick, not `# add the column`.
- Don't import from `app.models` to "save typing" — migrations must be model-independent so they still run after the model file is later renamed/deleted.
