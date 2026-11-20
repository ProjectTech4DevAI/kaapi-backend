"""Request-body examples for the Credentials routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

CREDENTIALS_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Store OpenAI and Langfuse credentials",
        {
            "is_active": True,
            "credential": {
                "openai": {"api_key": "sk-proj-xxxxxxxx"},
                "langfuse": {
                    "public_key": "pk-lf-xxxxxxxx",
                    "secret_key": "sk-lf-xxxxxxxx",
                    "host": "https://cloud.langfuse.com",
                },
            },
        },
    ),
}

CREDENTIALS_UPDATE_EXAMPLES: Examples = {
    "basic": (
        "Update the OpenAI credential",
        {
            "provider": "openai",
            "credential": {"api_key": "sk-proj-xxxxxxxx"},
            "is_active": True,
        },
    ),
}
