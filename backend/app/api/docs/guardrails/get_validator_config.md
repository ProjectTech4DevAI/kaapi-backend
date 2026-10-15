Fetch a single validator config by id, scoped to the calling project.

The response is the stored row flattened together with its validator-specific tuning config, so it carries keys beyond the declared schema — `id`, `organization_id`, `project_id`, `created_at`, `updated_at`, and every tuning key for that validator type.

### Errors

- `404` — no such config, or it belongs to another project.
- `422` — `config_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
