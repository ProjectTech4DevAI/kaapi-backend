Fetch a single ban list by id.

Readable if the list belongs to the calling project, or if it belongs to another tenant and is marked `is_public`.

### Errors

- `403` — the list belongs to another tenant and is not public.
- `404` — no such ban list.
- `422` — `ban_list_id` is not a valid UUID.
- `502` — the guardrails service is unreachable or returned a non-JSON body.
