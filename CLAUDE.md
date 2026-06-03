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

# Generate database migration (rev-id should be latest existing revision ID + 1)
alembic revision --autogenerate -m "Description" --rev-id 040

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

### Type Hints

Always add type hints to all function parameters and return values.

### Logging Format

Prefix all log messages with the function name in square brackets.

```python
logger.info(f"[function_name] Message {mask_string(sensitive_value)}")
```

### Database Column Comments

Use sa_column_kwargs["comment"] to describe database columns, especially when the purpose isn’t obvious. This helps non-developers understand column purposes directly from the database schema:

```python
field_name: int = Field(
    foreign_key="table.id",
    nullable=False,
    ondelete="CASCADE",
    sa_column_kwargs={"comment": "What this column represents"}
)
```

Prioritize comments for:
- Columns with non-obvious purposes
- Status/type fields (document valid values)
- JSON/metadata columns (describe expected structure)
- Foreign keys (clarify the relationship)

### Endpoint Documentation

Load Swagger descriptions from external markdown files instead of inline strings:

```python
@router.post(
    "/endpoint",
    description=load_description("domain/action.md"),
    response_model=APIResponse[ResponseModel],
)
```

Store documentation files in `backend/app/api/docs/<domain>/<action>.md`
