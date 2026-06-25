# Cross-cutting conventions (every layer)

The rules that apply across **every** layer of kaapi-backend, regardless of which file you're
editing. This is the authoritative source; CLAUDE.md carries a terse summary that defers here,
and the layer agents/conventions reference it rather than restating it.

## Rules

- **Type hints** on every parameter and return value. `-> Any` is not an annotation — narrow it
  or drop it. Use `|` unions (`str | None`), not `Optional[str]`.
- **Logging prefix:** every log line starts with the function (or `ClassName.method`) name in
  square brackets:
  ```python
  logger.info(f"[function_name] Message | key: {value}")
  ```
  Mask secrets / PII with `mask_string` from `app.utils`; never log raw payloads that may carry
  sensitive data. Pick the log *level* by who is at fault, not merely "did it fail" — see
  `error-handling.md` for the fault-based level table.
- **`uv` is the runner**, not `pip`: `uv run pytest`, `uv run alembic ...`,
  `uv run pre-commit run --all-files`.
- **No magic values.** Extract repeated literals (provider names, status strings, route paths,
  magic numbers like `1_000_000`) to a constant / `Enum` / settings. Operational config (worker
  counts, model names, timeouts, retry counts) goes to env / config, not hardcoded.
- **Comments explain *why*, not *what*.** Don't restate what the code already says
  (`i += 1  # increment i`), don't narrate self-evident lines, and don't pad docstrings /
  migration descriptions with obvious recaps. A comment earns its place only by adding
  non-obvious context — rationale, a gotcha, a link, a constraint. When in doubt, delete it.
- **Naming:** `list_*` for plural fetch, `get_*` for singletons (verb plurality matches the
  return shape); snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants; `Enum`
  suffix on enum classes. No leftover names from copy-pasting a sibling file.
- **Timestamps** are `inserted_at` / `updated_at`, never `created_at` (migration 060 renamed the
  legacy stragglers — do not reintroduce them).

## Layering (which layer may do what)

- **Routes** are thin: parse/validate input, call a service, wrap the result in `APIResponse[T]`.
  `HTTPException` belongs here.
- **Services** hold business logic and orchestration — the only layer that combines DB access
  (via `crud/`) with external HTTP (OpenAI, Langfuse, S3, webhooks). `HTTPException` is
  acceptable here for orchestration.
- **CRUD** is DB-only: returns data / `None` / raises domain errors. **Never** `HTTPException`,
  **never** third-party network calls.
- **Models** are leaf nodes: pure data shape, no imports from `fastapi`, `app.crud`, or
  `app.services`.

Per-layer detail lives in the matching convention doc (`model.md`, `crud.md`, `service.md`,
`route.md`, `migration.md`, `celery.md`, `test.md`, `error-handling.md`) — load the one for the
layer you're touching.
