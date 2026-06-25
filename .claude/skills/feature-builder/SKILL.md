---
name: feature-builder
description: Use when executing a kaapi-backend feature from its SRD (or a direct request) — writing models, CRUD, services, routes, Alembic migrations, Celery tasks, or tests. Walks the model→crud→service→route spine, loading each layer's house conventions before writing it.
---

# Feature Builder

Build a feature by walking the dependency spine and loading the house conventions for each layer
*before* you write it. This runs **inline in the current context**. For a large feature you may
dispatch a general-purpose subagent per phase to keep context lean — pass forward only the
artifacts (function signatures, file paths, the next migration rev-id), never re-derive prior
reasoning.

## Workflow

1. **Start from the SRD, then scope the layers.** Read `features/<slug>/SRD.md` — its endpoints,
   schema, execution flow, and **Impact / Blast Radius** section define *what* to build and what's
   in / out of scope. (No SRD? Build from the user's request.) Then decide which layers the change
   actually touches: a new entity touches all of model → crud → service → route; a new endpoint
   over existing data may be only service + route; a query-only change may be crud + route. Build
   only what's needed.

2. **Load conventions as you go.** Use the **`backend-conventions`** skill: Read
   `.claude/conventions/cross-cutting.md` first, then Read each layer's doc *right before* you
   write that layer (lazy-load — skip docs for layers the feature doesn't touch).

3. **Walk the spine in dependency order: model → crud → service → route.** Each downstream layer
   depends on the one above (route calls service, service calls crud, crud uses model), so never
   build out of order. Build them all in this one pass — no per-layer handoff.

4. **Migration.** If you added/changed/renamed a model field, write the Alembic migration
   (`migration.md`). Rev-id = highest `NNN_*.py` in `backend/app/alembic/versions/` + 1,
   zero-padded — recompute from the directory, don't guess.

5. **Async / external SDKs.** Background work → write the Celery task (`celery.md`). Any code that
   calls an external SDK or makes raw HTTP → apply the error-handling pattern
   (`error-handling.md`).

6. **Tests.** Write tests (`test.md`) and run the relevant subset
   (`uv run pytest backend/app/tests/... -k <name> -x`). Fix real failures before moving on.

7. **Review before declaring done.** Run **`/pr-review`** on your diff (or apply its checklist) to
   catch convention, security, and correctness issues before declaring the feature complete.

## Rules

- **Single-layer change?** Load just that one convention doc and write it — don't walk the whole
  spine.
- **No model/schema change?** Skip the migration step.
- Cross-cutting rules (type hints, `[function_name]` logging + secret masking, `inserted_at` /
  `updated_at`, no magic values, multi-tenant `organization_id` / `project_id`) apply at **every**
  layer — see `cross-cutting.md`.
- **One summary at the end** (not one per layer): the layers built + key signatures (model
  variants; crud/service/route function signatures + paths); any new `Permission` enum value,
  domain exception, or `.env.example` / settings key the user must add; whether a migration is
  needed and its rev-id; what the tests cover.
