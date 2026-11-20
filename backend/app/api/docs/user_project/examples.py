"""Request-body examples for the User Project routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

USER_PROJECT_ADD_EXAMPLES: Examples = {
    "basic": (
        "Add users to a project",
        {
            "organization_id": 1,
            "project_id": 1,
            "users": [
                {"email": "jane.doe@example.com", "full_name": "Jane Doe"},
                {"email": "ravi.kumar@example.com", "full_name": "Ravi Kumar"},
            ],
        },
    ),
}
