# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

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

Layer-specific conventions live in `.claude/agents/*.md` and are enforced by the matching specialist subagent (e.g., `route-writer` for `app/api/routes/`, `model-writer` for `app/models/`, `migration-writer` for alembic). CLAUDE.md only covers rules that apply across every layer.

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

When working in a specific layer, the matching agent under `.claude/agents/` handles the layer's conventions automatically. Pick by layer, or just describe the task and let the main agent route:

| Agent | Layer |
|---|---|
| `feature-builder` | Full feature spanning `models` → `crud` → `services` → `api/routes` (the build spine, one context) |
| `route-writer` | `app/api/routes/` (single-layer edits) |
| `crud-writer` | `app/crud/` (single-layer edits) |
| `service-writer` | `app/services/` (single-layer edits) |
| `model-writer` | `app/models/` (single-layer edits) |
| `migration-writer` | `app/alembic/versions/` |
| `celery-task-writer` | `app/celery/tasks/` |
| `test-writer` | `app/tests/` |
| `convention-reviewer` | Cross-cutting pre-commit gate (mirrors `/pr-review`) |

### Build a feature as a 4-context pipeline

To keep each context window lean (heavy file I/O degrades performance), **build a multi-layer feature as four sequential subagent contexts, not inline.** Launch each phase with the Agent tool — each runs in its own context and returns only a summary, so the orchestrator stays small. The phases are a dependency chain, so run them **in order**, passing only the *artifacts* forward (signatures, file paths, the next migration rev-id), never re-deriving prior reasoning:

| # | Context | Agent | Consumes |
|---|---|---|---|
| 1 | schema + code-spine | `feature-builder` | the feature request |
| 2 | migration | `migration-writer` | phase 1's model changes + next rev-id |
| 3 | test | `test-writer` | phase 1's signatures (+ which HTTP boundaries to mock) |
| 4 | review | `convention-reviewer` | the full diff |

Rules of thumb:
- **Run them sequentially**, not in parallel — phase N depends on phase N-1's output.
- **Single-layer change?** Skip the pipeline; delegate to the one matching standalone agent (e.g. `crud-writer`) directly.
- **No model/schema change?** Skip phase 2 (`migration-writer`).
- `feature-builder` and the standalone layer agents share one source of truth — the convention docs in `.claude/conventions/{model,crud,service,route}.md` — so output never drifts between them.
