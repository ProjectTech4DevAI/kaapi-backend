"""Request-body examples for the Collections routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

COLLECTION_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Create a collection from uploaded documents",
        {
            "name": "Program FAQs",
            "description": "FAQ and policy documents for the helpline assistant",
            "documents": [
                "6f1c2b9e-4a3d-4c8e-9b7a-2d5e8f1a3c47",
                "a2d4e6f8-1b3c-4d5e-8f9a-0b1c2d3e4f5a",
            ],
            "provider": "openai",
            "callback_url": "https://example.com/webhooks/collections",
        },
    ),
}

COLLECTION_UPDATE_EXAMPLES: Examples = {
    "rename": (
        "Update name and description",
        {
            "name": "Program FAQs 2026",
            "description": "Updated FAQ and policy documents",
        },
    ),
}

COLLECTION_DELETE_EXAMPLES: Examples = {
    "with_callback": (
        "Delete and get notified on a callback URL",
        {"callback_url": "https://example.com/webhooks/collections"},
    ),
}
