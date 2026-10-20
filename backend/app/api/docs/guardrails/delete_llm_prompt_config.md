Delete a stored LLM prompt config permanently, scoped to the calling project.

Validator configs that still reference the deleted prompt will fail when the guardrails service next tries to resolve it. Repoint or remove them first.

Responds `200` with a confirmation body rather than `204`.

### Errors

- `404` — no such config, or it belongs to another project.
- `422` — `prompt_config_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
