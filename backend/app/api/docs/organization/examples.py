"""Request-body examples for the Organization routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

ORGANIZATION_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Create an organization",
        {"name": "Acme Health Foundation", "is_active": True},
    ),
}

ORGANIZATION_UPDATE_EXAMPLES: Examples = {
    "rename": ("Rename the organization", {"name": "Acme Health Foundation India"}),
}

ORGANIZATION_DELETE_EXAMPLES: Examples = {
    "soft_delete": ("Soft delete (default)", {"hard_delete": False}),
}
