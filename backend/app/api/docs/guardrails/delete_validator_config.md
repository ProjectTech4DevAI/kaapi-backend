Delete a validator config permanently, scoped to the calling project.

Guardrails runs that still pass the deleted `validator_config_id` will no longer resolve it. Remove the id from your `POST /guardrails` calls first.

Responds `200` with a confirmation body rather than `204`.

### Errors

- `404` — no such config, or it belongs to another project.
- `422` — `config_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
