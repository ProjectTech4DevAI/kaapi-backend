"""Request-body examples for the Model Config routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

_MODEL_CREATE_BODY = {
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "completion_type": ["text"],
    "config": {
        "temperature": {
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 2.0,
            "description": "Controls randomness.",
        }
    },
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"],
    "pricing": {
        "response": {
            "input_token_cost": 0.15,
            "output_token_cost": 0.6,
        },
        "batch": {
            "input_token_cost": 0.075,
            "output_token_cost": 0.3,
        },
    },
    "is_active": True,
}

MODEL_CREATE_EXAMPLES: Examples = {
    "single": ("Register one model", _MODEL_CREATE_BODY),
    "bulk": (
        "Register several models at once",
        [
            _MODEL_CREATE_BODY,
            {
                "provider": "openai",
                "model_name": "gpt-4o-mini-transcribe",
                "completion_type": ["stt"],
                "input_modalities": ["AUDIO"],
                "output_modalities": ["TEXT"],
            },
        ],
    ),
}

MODEL_UPDATE_EXAMPLES: Examples = {
    "full_update": (
        "Update capabilities, pricing and status",
        {
            "completion_type": ["text", "stt"],
            "config": {
                "temperature": {
                    "type": "float",
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "description": "Controls randomness.",
                }
            },
            "input_modalities": ["TEXT", "AUDIO"],
            "output_modalities": ["TEXT"],
            "pricing": {
                "response": {"input_token_cost": 0.5, "output_token_cost": 2.0},
                "batch": {"input_token_cost": 0.25, "output_token_cost": 1.0},
                "audio": {"input_token_cost": 1.0, "output_token_cost": 2.0},
            },
            "is_active": True,
        },
    ),
}

MODEL_BULK_UPDATE_EXAMPLES: Examples = {
    "bulk_update": (
        "Update pricing and deactivate a model",
        [
            {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "pricing": {
                    "response": {
                        "input_token_cost": 0.15,
                        "output_token_cost": 0.6,
                    }
                },
            },
            {
                "provider": "google",
                "model_name": "gemini-2.0-flash",
                "is_active": False,
            },
        ],
    ),
}
