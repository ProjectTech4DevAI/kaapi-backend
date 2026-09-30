Update a ban list. Fields you omit are left unchanged.

`banned_words` is replaced wholesale, not merged — send the full list you want stored.

### Request

```json
{
  "description": "Updated description",
  "banned_words": ["slur_a", "slur_b", "slur_c"]
}
```

All five fields (`name`, `description`, `banned_words`, `domain`, `is_public`) are optional and follow the same constraints as on create.

### Errors

- `400` — the update collides with an existing ban list.
- `403` — the list belongs to another tenant. Public lists are readable but not writable across tenants.
- `404` — no such ban list.
- `422` — the body failed validation, or `ban_list_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
