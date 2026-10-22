Update a validator config's base fields. Fields you omit are left unchanged.

### Request

```json
{"stage": "output", "is_enabled": false}
```

Only `name`, `type`, `stage`, `on_fail_action` and `is_enabled` can be patched.

### Notes

- **Validator-specific tuning cannot be changed here.** The guardrails service rejects any key outside the five base fields, so changing something like `threshold` or `entity_types` means deleting the config and recreating it.
- Because `POST /guardrails` resolves validators by id at run time, an update takes effect on the next run — there is no versioning.

### Errors

- `400` — the new `name` collides with an existing config in the project.
- `404` — no such config, or it belongs to another project.
- `422` — the guardrails service rejected the body, e.g. a key outside the five base fields. A malformed `config_id` is rejected here, before the call is forwarded.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
