# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Kaapi is an AI platform built with FastAPI and PostgreSQL, containerized with Docker. It provides AI capabilities including multi-provider LLM access (OpenAI, Anthropic, Google, plus speech providers like Sarvam/ElevenLabs — orchestrated via `litellm`), fine-tuning, document processing, collection management, and evaluation runs. Async work runs on Celery (RabbitMQ broker, Redis backend); observability via Langfuse, OpenTelemetry, and Sentry.

## Key Commands

### Development

**The Python project root is `backend/`, not the repo root.** Every command below assumes `cd backend` first (`pyproject.toml`, `uv.lock`, `alembic.ini`, and `scripts/` all live there). Paths like `app/...` in this doc are relative to `backend/`.

```bash
# Bring up the full stack (Postgres, RabbitMQ, Redis, backend, celery, Adminer, Flower) — from repo root
docker compose watch

# --- the rest run from backend/ ---
cd backend

# Start development server with auto-reload (stop the docker `backend` service first)
fastapi run --reload app/main.py

# Run the Celery worker locally (separate terminal)
uv run celery -A app.celery.celery_app worker --loglevel=info

# Lint / format (also wired into pre-commit)
bash scripts/lint.sh      # ruff check + mypy strict
bash scripts/format.sh    # ruff format + import sort
uv run pre-commit run --all-files

# Generate database migration.
# Compute <next_rev_id> at runtime as the latest existing revision ID + 1,
# zero-padded to 3 digits (check the highest NNN in app/alembic/versions/NNN_*.py).
uv run alembic revision --autogenerate -m "Description" --rev-id <next_rev_id>
uv run alembic upgrade head

# Seed database with test data
uv run python -m app.seed_data.seed_data

# Internal admin/data CLI (entrypoint: app.cli.main:cli)
uv run ai-cli --help
```

### Testing

Tests use `.env.test` for environment-specific configuration (env var `ENVIRONMENT=testing`). Run against a **real Postgres** — the suite runs `tests_pre_start.py` + `alembic upgrade head` before pytest; do not mock the DB session.

```bash
# Full suite with coverage, from backend/ (applies test migrations, then pytest)
uv run bash scripts/tests-start.sh

# A single test (after migrations are applied), from backend/
uv run pytest app/tests/path/to/test_file.py::test_name -x

# Dockerized run (from repo root): builds, brings up the stack, runs the suite, tears down
bash scripts/test.sh
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

## Environment Configuration

The application uses different environment files:
- `.env` - Application environment configuration (use `.env.example` as template)
- `.env.test` - Test environment configuration


## Testing Strategy

- Tests located in `app/tests/`
- Factory pattern for test fixtures
- Automatic coverage reporting

## Code Standards

- Python 3.12+ with type hints (`requires-python = ">=3.12"`; mypy `strict`, ruff target `py312`)
- Pre-commit hooks for linting and formatting

## Coding Conventions

Layer-specific conventions live in **`.claude/conventions/*.md`** — one doc per layer (`model`, `crud`, `service`, `route`, `migration`, `celery`, `test`, `error-handling`) plus `cross-cutting.md`. The **`backend-conventions`** skill indexes them, and the `feature-builder` skill loads each layer's doc when *building* a feature. Same source of truth, so design and code never drift.

### Cross-cutting rules

These are a terse summary for quick reference; `.claude/conventions/cross-cutting.md` is the authoritative, fuller version (update that file, not just this list).

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

## Skills (the feature lifecycle)

Work is driven by skills, not specialist subagents. The lifecycle:

| Stage | Skill | Output |
|---|---|---|
| Product spec | `start-prd` | `features/<slug>/PRD.md` |
| Software spec | `srd-creator` | `features/<slug>/SRD.md` (includes blast-radius impact analysis via `docs/domain-map.md`) |
| Execute SRD (build) | `feature-builder` | the code: model → crud → service → route, then migration / celery / tests |
| Review | `/pr-review` | convention / security / correctness gate on the diff |

The `backend-conventions` skill is the conventions index that `feature-builder` loads per layer while building.

### Building a feature

The `feature-builder` skill walks the dependency spine **model → crud → service → route** in the current context, loading each layer's convention doc (via `backend-conventions`) right before writing that layer, then the migration / Celery / tests as needed, and finishes by running `/pr-review` on the diff. Only build the layers the feature touches; a single-layer change just loads that one doc.

For a large feature where context size is a concern, you may dispatch a **general-purpose subagent per phase** (schema-spine → migration → tests) and pass only the artifacts forward (signatures, file paths, next migration rev-id) — the `feature-builder` skill works the same whether run inline or inside a dispatched subagent.
