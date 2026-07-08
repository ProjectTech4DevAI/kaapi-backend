Update project-level settings.

Patches the `settings` JSONB of the project bound to the authenticating organization
API key. Only the keys provided in the request body are changed; existing keys are kept.

**Settings**

- `tracing` (bool): enable/disable Langfuse tracing for this project. Off by default to
  conserve the org's Langfuse rate-limit/credit budget. Gates tracing for both the
  response path and evaluations; when off, evaluations fall back to cosine-only scoring.

**Scope:** requires an organization API key with project access.
