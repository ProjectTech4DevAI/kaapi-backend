#!/bin/bash
set -e
set -x

# Parallel run with coverage; pytest-cov merges coverage across xdist workers
pytest -n auto \
    --cov=app \
    --cov-report=term-missing \
    --cov-report="html:htmlcov" \
    --cov-report=xml
