Update settings for a project by ID.

Patches the `settings` JSONB of the project identified by the path `project_id`. Only the
keys provided in the request body are changed; existing keys are kept.

**Settings**

- `tracing` (bool): enable/disable Langfuse tracing for this project. Off by default to
  conserve the org's Langfuse rate-limit/credit budget. Gates tracing for both the
  response path and evaluations; when off, evaluations fall back to cosine-only scoring.

**Scope:** superusers may patch any project across organizations. A project-scoped key may
patch only its own bound project; targeting any other `project_id` returns 403.
