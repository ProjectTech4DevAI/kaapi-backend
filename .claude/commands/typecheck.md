---
description: Run pyright type checking on Python files and triage the errors.
---

Run pyright over the backend and report type errors in this FastAPI + SQLModel service.

## Run

From the repo root:

```bash
cd backend && bash scripts/pyright.sh $ARGUMENTS
```

- No `$ARGUMENTS` → checks all of `app/` (config in `backend/pyproject.toml` `[tool.pyright]`).
- `$ARGUMENTS` → narrow to those paths/files, e.g. `/typecheck app/services/response`.
- Pyright runs via `uvx` (ephemeral) so it never mutates `uv.lock`.

## Report

1. Group errors by file, most-affected first.
2. For each: `file:line` — the error, then the concrete fix (narrow the type, add an annotation, guard the `None`, etc.).
3. Separate real type bugs from noise (missing third-party stubs, `# type: ignore` candidates). Flag which is which.
4. Do NOT auto-edit code — surface findings and let the user pick what to fix. If asked to fix, follow the layer conventions in `.claude/conventions/`.
