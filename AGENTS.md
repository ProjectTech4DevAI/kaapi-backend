# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

Kaapi is an AI platform built with FastAPI and PostgreSQL, containerized with Docker. It provides AI capabilities including OpenAI assistants, fine-tuning, document processing, and collection management.

## Key Commands

### Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Start development server with auto-reload
fastapi run --reload app/main.py

# Run pre-commit hooks
uv run pre-commit run --all-files

# Generate database migration.
# Compute <next_rev_id> at runtime as the latest existing revision ID + 1,
# zero-padded to 3 digits (check the highest NNN in app/alembic/versions/NNN_*.py).
alembic revision --autogenerate -m "Description" --rev-id <next_rev_id>

# Seed database with test data
uv run python -m app.seed_data.seed_data
```

### Testing

Tests use `.env.test` for environment-specific configuration.

```bash
# Run test suite
uv run bash scripts/tests-start.sh
```

## Architecture

### Backend Structure

The backend follows a layered architecture located in `backend/app/`:

- **Models** (`models/`): SQLModel entities representing database tables and domain objects

- **CRUD** (`crud/`): Database access layer for all data operations

- **Routes** (`api/`): FastAPI REST endpoints organized by domain

- **Core** (`core/`): Core functionality and utilities
  - Configuration and settings
  - Database connection and session management
  - Security (JWT, password hashing, API keys)
  - Cloud storage (`cloud/storage.py`)
  - Document transformation (`doctransform/`)
  - Fine-tuning utilities (`finetune/`)
  - Langfuse observability integration (`langfuse/`)
  - Exception handlers and middleware

- **Services** (`services/`): Business logic services
  - Response service (`response/`): OpenAI Responses API integration, conversation management, and job execution

- **Celery** (`celery/`): Asynchronous task processing with RabbitMQ and Redis
  - Task definitions (`tasks/`)
  - Celery app configuration with priority queues
  - Beat scheduler and worker configuration


### Authentication & Security

- JWT-based authentication
- API key support for programmatic access
- Organization and project-level permissions

## Codebase Knowledge (wiki)

`docs/wiki/INDEX.md` routes to per-module knowledge pages (routes, tables, models, services per domain) and `docs/wiki/domain-map.md` (entity graph + blast-radius procedure). Load discipline:

- Starting feature or planning work in a domain → read INDEX + that one `docs/wiki/modules/*.md` page before exploratory greps.
- Adding a table or config shape → check `domain-map.md` first; an existing entity/flow may already cover it.
- Deep design rationale → follow the module page's link into `docs/architecture/*.md`; never bulk-load those.
- **Maintenance rule:** a change to a module's routes/tables/models/services updates that module's wiki page (and `domain-map.md` if entities/edges changed) in the same PR.

## Environment Configuration

The application uses different environment files:
- `.env` - Application environment configuration (use `.env.example` as template)
- `.env.test` - Test environment configuration


## Testing Strategy

- Tests located in `app/tests/`
- Factory pattern for test fixtures
- Automatic coverage reporting

## Code Standards

- Python 3.11+ with type hints
- Pre-commit hooks for linting and formatting

## Coding Conventions

Layer conventions live in `.Codex/conventions/{model,crud,service,route,migration,celery}.md` and are applied by the `senior-engineer` subagent; the `test-writer` agent carries its own conventions in `.Codex/agents/*.md`. AGENTS.md only covers rules that apply across every layer.

### Cross-cutting rules

- **Type hints** on every parameter and return value. `-> Any` is not an annotation — narrow it or drop it.
- **Logging prefix:** every log line starts with the function name in square brackets.
  ```python
  logger.info(f"[function_name] Message | key: {value}")
  ```
- **`uv` is the runner**, not `pip`. Examples: `uv run pytest`, `uv run alembic ...`, `uv run pre-commit run --all-files`.
- **No magic values** in code — extract repeated literals to constants / `Enum` / settings.
- **Comments explain *why*, not *what*.** Don't restate what the code already says (`i += 1  # increment i`), don't narrate self-evident lines, and don't pad docstrings/migration descriptions with obvious recaps of the operations. A comment earns its place only by adding non-obvious context — rationale, a gotcha, a link, a constraint. When in doubt, delete it; clear code needs fewer comments, not more.
- **Naming:** `list_*` for plural fetch, `get_*` for singletons; snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants; `Enum` suffix on enum classes.
- **Timestamps** are `inserted_at` / `updated_at` (not `created_at`).

## Specialist subagents

When working in a specific layer, the matching agent under `.Codex/agents/` handles the layer's conventions automatically. Pick by layer, or just describe the task and let the main agent route:

| Agent | Layer |
|---|---|
| `senior-engineer` | `app/models/`, `app/crud/`, `app/services/`, `app/api/routes/`, `app/alembic/versions/`, `app/celery/tasks/` — any single-layer edit or a full feature walking the spine plus its migration and Celery task, one context |
| `test-writer` | `app/tests/` |

Standardized provider/SDK exception handling is a cross-cutting convention (`.Codex/conventions/error-handling.md`), applied by `senior-engineer` when it writes service/crud call sites — not a separate agent. Convention reviews are handled by the `/pr-review` command, also not a subagent.

### Build a feature as a 2-context pipeline

To keep each context window lean (heavy file I/O degrades performance), **build a multi-layer feature as sequential subagent contexts, not inline.** Launch each phase with the Agent tool — each runs in its own context and returns only a summary, so the orchestrator stays small. The phases are a dependency chain, so run them **in order**, passing only the *artifacts* forward (signatures, file paths), never re-deriving prior reasoning:

| # | Context | Agent | Consumes |
|---|---|---|---|
| 1 | schema + code-spine + migration + Celery task | `senior-engineer` | the feature request |
| 2 | test | `test-writer` | phase 1's signatures (+ which HTTP boundaries to mock) |

Then run `/pr-review` on the full diff before committing.

Rules of thumb:
- **Run them sequentially**, not in parallel — phase 2 depends on phase 1's signatures.
- **Single-layer change?** Skip the pipeline; let `senior-engineer` build just the one layer.
- `senior-engineer` builds the model → crud → service → route spine *and* the migration a schema change needs *and* the Celery task background work needs — all in phase 1, reading the convention docs in `.Codex/conventions/{model,crud,service,route,migration,celery}.md` as the single source of truth for each layer.
