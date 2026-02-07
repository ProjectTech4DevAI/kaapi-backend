#!/bin/bash
set -e

uv run alembic upgrade head

exec uv run uvicorn app.main:app --host 0.0.0.0 --port 80 --workers 4
