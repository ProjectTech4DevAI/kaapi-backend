---
name: backend-conventions
description: Use when planning, designing, building, or reviewing kaapi-backend code and you need the house conventions for a layer — models, CRUD, services, routes, Alembic migrations, Celery tasks, tests, error handling, or the cross-cutting rules. Loads the authoritative per-layer convention docs on demand.
---

# Backend conventions

The single index of kaapi-backend's code conventions. Each layer's authoritative rules live in a
doc under `.claude/conventions/`. This skill maps the layer you're touching to its doc so you
load only what you need.

## How to use this

1. Identify which layers your task touches (a full feature touches several; a single edit may
   touch one).
2. **`Read` the cross-cutting doc first**, then **`Read` the doc for each layer you will write or
   review.** Do this *before* writing or reviewing that layer's code — these docs are the source
   of truth, not background reading. When planning a feature, read the docs for every layer the
   plan's LLD will specify.
3. Apply each doc's rules to the task. If a doc names a canonical reference file in the repo
   (e.g. `app/models/user.py`), read that too before writing.

## Layer → convention doc

| Layer / concern | Path under `app/` | Convention doc | Covers |
|---|---|---|---|
| **Cross-cutting** (always) | every file | `.claude/conventions/cross-cutting.md` | type hints, logging prefix + masking, `uv`, no magic values, comments, naming, timestamps, layering boundaries |
| **Models** | `app/models/` | `.claude/conventions/model.md` | Base/Create/Update/Public split, `sa_column_kwargs` comments, FK indexes, Enum naming, first-class columns over JSON |
| **CRUD** | `app/crud/` | `.claude/conventions/crud.md` | DB-only access, no `HTTPException`, eager loading to avoid N+1, logging style |
| **Services** | `app/services/` | `.claude/conventions/service.md` | orchestration (DB + external HTTP), SSRF guards, narrow try blocks, domain-error translation |
| **Routes** | `app/api/routes/` | `.claude/conventions/route.md` | `response_model=APIResponse[T]`, `load_description(...)`, permission deps, status codes, swagger markdown |
| **Migrations** | `app/alembic/versions/` | `.claude/conventions/migration.md` | `--rev-id` discipline, reversible downgrades, in-upgrade backfills, FK indexes, CONCURRENTLY constraints |
| **Celery tasks** | `app/celery/tasks/` | `.claude/conventions/celery.md` | queue/priority, retry policy, idempotency, OTel propagation, `gevent_timeout` |
| **Tests** | `app/tests/` | `.claude/conventions/test.md` | factory pattern, transactional `db` fixture, real-DB (no mocked sessions), behavior asserts |
| **Error handling** | provider wrappers, SDK-calling CRUD | `.claude/conventions/error-handling.md` | `[KAAPI]`/`[<PROVIDER>]` source tags, status→message templates, fault-based log levels |

## Who else uses this

- `feature-builder` loads each layer's doc right before writing that layer while executing a
  feature's SRD — these docs are the single source of truth for code style.
- `/pr-review` is the review gate; it carries its own checklist mirroring these docs, so when a
  convention doc changes, update that checklist to match.
