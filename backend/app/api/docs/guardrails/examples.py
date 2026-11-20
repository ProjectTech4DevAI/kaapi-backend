"""Request-body examples for the Guardrails routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

GUARDRAILS_APPLY_EXAMPLES: Examples = {
    "pii_and_stored_validator": (
        "Remove PII and run a stored validator",
        {
            "text": "Hi, I am Priya, my phone number is 98765 43210. Is fever normal after vaccination?",
            "config": [
                {"type": "pii_remover", "on_fail": "fix"},
                {"validator_config_id": "8c1d2e3f-4a5b-4c6d-9e7f-0a1b2c3d4e5f"},
            ],
            "callback_url": "https://example.com/webhooks/guardrails",
            "request_metadata": {"message_id": "msg-1042"},
        },
    ),
}
