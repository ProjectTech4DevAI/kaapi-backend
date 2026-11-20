"""Request-body examples for the Projects routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

PROJECT_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Create a project in an organization",
        {
            "name": "WhatsApp Helpline Bot",
            "description": "AI assistant answering maternal health questions over WhatsApp",
            "is_active": True,
            "organization_id": 1,
        },
    ),
}

PROJECT_UPDATE_EXAMPLES: Examples = {
    "description": (
        "Update only the description",
        {
            "description": "AI assistant answering maternal and child health questions over WhatsApp"
        },
    ),
}

PROJECT_SETTINGS_UPDATE_EXAMPLES: Examples = {
    "enable_tracing": ("Turn on Langfuse tracing", {"tracing": True}),
}

PROJECT_DELETE_EXAMPLES: Examples = {
    "soft_delete": ("Soft delete (default)", {"hard_delete": False}),
}
