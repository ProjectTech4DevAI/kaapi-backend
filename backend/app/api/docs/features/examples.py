"""Request-body examples for the Feature Flags routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

FEATURE_FLAG_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Enable a flag for a project",
        {"key": "ASSESSMENT", "project_id": 1, "enabled": True},
    ),
}

FEATURE_FLAG_UPDATE_EXAMPLES: Examples = {
    "disable": (
        "Disable a flag for a project",
        {"key": "ASSESSMENT", "project_id": 1, "enabled": False},
    ),
}

FEATURE_FLAG_DELETE_EXAMPLES: Examples = {
    "basic": ("Remove a flag from a project", {"key": "ASSESSMENT", "project_id": 1}),
}
