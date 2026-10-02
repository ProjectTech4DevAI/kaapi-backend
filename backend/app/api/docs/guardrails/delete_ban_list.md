Delete a ban list permanently. Restricted to the owning project.

Validator configs that still reference the deleted list by `ban_list_id` are not cleaned up; they will fail when the guardrails service next tries to resolve the list.

Responds `200` with a confirmation body rather than `204`.

### Errors

- `403` — the list belongs to another tenant.
- `404` — no such ban list.
- `422` — `ban_list_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
