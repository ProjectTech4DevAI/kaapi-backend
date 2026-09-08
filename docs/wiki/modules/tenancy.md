# Module: Tenancy (Users / Orgs / Projects / API Keys)

Identity and multi-tenant scoping. Permission details: `cross-cutting/auth.md`.

All paths relative to `backend/app/`.

## Routes
- `api/routes/login.py`, `api/routes/auth.py` — login/JWT
- `api/routes/users.py`, `api/routes/user_project.py` — users + membership
- `api/routes/organization.py`, `api/routes/project.py`
- `api/routes/api_keys.py`
- `api/routes/onboarding.py`
- `api/routes/private.py` — internal endpoints

## Tables (SQLModel)
| Table | Model |
|---|---|
| `user` (User) | `models/user.py` |
| `organization` (Organization) | `models/organization.py` |
| `project` (Project; FK → organization) | `models/project.py` |
| `user_project` (UserProject; user↔project membership) | `models/user_project.py` |
| `api_key` (APIKey; FK → org, project, user) | `models/api_key.py` |

## Services / CRUD
- `crud/user.py`, `crud/organization.py`, `crud/project.py`, `crud/user_project.py`, `crud/api_key.py`, `crud/auth.py`, `crud/onboarding.py`
- `services/auth.py`

## API contracts
- `OnboardingRequest.credentials` is a list of single-entry `{Provider: payload}`
  maps typed against the per-provider credential models in `core/providers.py`
  (see `modules/platform.md`). Unsupported providers, non-object payloads and
  missing required fields are rejected at the request boundary (422).
- `ProjectPublic.settings` is `dict[str, JsonValue]` — free-form JSONB whose
  writable keys are defined by `ProjectSettingsUpdate` (`PATCH
  /projects/settings` for the key's own project; `PATCH
  /projects/{project_id}/settings` for superusers targeting any project),
  currently just `tracing`.
