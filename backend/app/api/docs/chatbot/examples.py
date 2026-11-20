"""Request-body examples for the Chatbot routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

CHATBOT_CREATE_EXAMPLES: Examples = {
    "basic": ("Create a chatbot", {"name": "Beneficiary Onboarding Bot"}),
}

CHATBOT_UPDATE_EXAMPLES: Examples = {
    "rename": ("Rename a chatbot", {"name": "Beneficiary Onboarding Bot v2"}),
}

CHATBOT_FLOW_SAVE_EXAMPLES: Examples = {
    "onboarding_flow": (
        "Save an input -> onboarding -> output flow",
        {
            "nodes": [
                {
                    "id": "input-1",
                    "type": "input",
                    "position": {"x": 0, "y": 100},
                    "data": {"label": "User message"},
                },
                {
                    "id": "onboarding-1",
                    "type": "onboarding",
                    "position": {"x": 250, "y": 100},
                    "data": {
                        "label": "Beneficiary Info Extractor",
                        "config_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "config_version": 1,
                        "prompt": "Collect the user's name and city.",
                    },
                },
                {
                    "id": "output-1",
                    "type": "output",
                    "position": {"x": 500, "y": 100},
                    "data": {"label": "Bot reply"},
                },
            ],
            "edges": [
                {
                    "id": "e-input-onboarding",
                    "source": "input-1",
                    "target": "onboarding-1",
                },
                {
                    "id": "e-onboarding-output",
                    "source": "onboarding-1",
                    "target": "output-1",
                },
            ],
        },
    ),
}

CHATBOT_SESSION_CREATE_EXAMPLES: Examples = {
    "basic": ("Start a session for a contact", {"contact_ref": "919876543210"}),
}

CHATBOT_SESSION_MESSAGE_EXAMPLES: Examples = {
    "with_webhook": (
        "Send a message and get the reply on a webhook",
        {
            "message": "Hi, my name is Priya and I live in Pune.",
            "webhook_url": "https://example.com/webhooks/chatbot-reply",
        },
    ),
}
