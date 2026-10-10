Register a validator configuration — a named, reusable validator setup that `POST /guardrails` applies by id.

This is the object you point at from a guardrails run: create one here, then pass its `id` as a `validator_config_id` in the `config` array of `POST /guardrails`.

### Request

The body is a fixed set of base fields **plus** whatever tuning keys the chosen validator type accepts. Call `GET /guardrails` to discover the accepted keys per type.

```json
{
  "name": "PII Redaction Input",
  "type": "pii_remover",
  "stage": "input",
  "on_fail_action": "fix",
  "is_enabled": true,
  "entity_types": ["PERSON", "PHONE_NUMBER", "IN_AADHAAR"],
  "threshold": 0.6
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | 5–225 chars. |
| `type` | enum | yes | One of the validator types from `GET /guardrails`. |
| `stage` | enum | yes | `input` or `output`. |
| `on_fail_action` | enum | no | `exception` \| `fix` \| `rephrase`. Defaults to `fix`. |
| `is_enabled` | boolean | no | Defaults to `true`. |
| *(extra keys)* | any | no | Validator-specific tuning, stored as the config blob. |

In the example above, `entity_types` and `threshold` are `pii_remover` tuning keys — they are not part of the base schema, which is why Swagger shows them as additional properties rather than named fields.

### Notes

- **Uniqueness is enforced on `name` alone**, scoped to the project. The same validator type may be registered many times under different names.
- `stage` is advisory. `POST /guardrails` routes on the text it is actually given, not on this field, so one config can serve both directions.
- Do **not** send `organization_id` or `project_id`. The tenant is derived from your authenticated context, and the guardrails service rejects those keys in the body.
- Responds `200`, not `201`.

### Errors

- `400` — a validator config with this name already exists in the project.
- `422` — the body failed validation, or it contained `organization_id`/`project_id`.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
