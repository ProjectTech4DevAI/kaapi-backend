# Cross-cutting: Auth & Permissions

All paths relative to `backend/app/`.

## Mechanisms
- **JWT** — user login sessions. Issued via `api/routes/login.py`; verification in `core/security.py`.
- **API keys** — programmatic access, scoped to (org, project, user). Table `api_key`; CRUD `crud/api_key.py`.
- **Org/project scoping** — every multi-tenant table carries `organization_id` + `project_id`; route dependencies resolve the caller's project and must filter every query by it.

## Key files
- `core/security.py` — JWT encode/verify, password hashing, API key handling
- `services/auth.py`, `crud/auth.py` — auth flows
- `models/user_project.py` — membership/role link between users and projects

## Rules for new endpoints
- Resolve (org, project) from the auth dependency, never from client-supplied IDs alone.
- Tenant isolation is per-query: a saved resource belonging to (org A, project A) must never resolve for another project.
