Fetch a single LLM prompt config by id, scoped to the calling project.

### Errors

- `404` — no such config, or it belongs to another project.
- `422` — `prompt_config_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
