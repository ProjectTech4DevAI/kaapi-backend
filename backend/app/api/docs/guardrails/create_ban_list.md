Create a ban list — a named set of words the `ban_list` validator redacts from text.

A ban list is stored by the guardrails service and scoped to the calling project. Reference it from a validator config by setting that config's `ban_list_id` to the `id` returned here.

### Request

```json
{
  "name": "Safety Banned Terms",
  "description": "Terms not allowed for this tenant policy",
  "domain": "abuse",
  "is_public": false,
  "banned_words": ["slur_a", "slur_b"]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | 1–100 chars. Must be unique for the project. |
| `description` | string | yes | 1–500 chars. |
| `banned_words` | string[] | yes | Up to 1000 entries, each 1–100 chars. |
| `domain` | string | yes | Free-form grouping label used to filter on list. |
| `is_public` | boolean | no | Defaults to `false`. |

### Notes

- `is_public: true` makes the list readable by other tenants. Updating and deleting stay restricted to the owning project regardless.
- The response is `200`, not `201` — the guardrails service does not use `201`.

### Errors

- `400` — a ban list with this configuration already exists.
- `422` — the body failed validation.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
