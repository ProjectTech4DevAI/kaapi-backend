"""Request-body examples for the Credentials routes.


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
