"""Request-body examples for the Config routes.


"""

from app.api.openapi_docs import Examples

CONFIG_CREATE_EXAMPLES: Examples = {
    "text_completion": (
        "Create an OpenAI text-completion config",
        {
            "name": "helpline-assistant",
            "description": "Answers beneficiary questions about the program",
            "config_blob": {
                "completion": {
                    "provider": "openai",
                    "type": "text",
                    "params": {
                        "model": "gpt-4o-mini",
                        "instructions": "You are a helpful assistant. Answer briefly and politely.",
                        "temperature": 0.3,
                        "max_output_tokens": 1024,
                    },
                }
            },
            "commit_message": "Initial configuration",
        },
    ),
}

CONFIG_UPDATE_EXAMPLES: Examples = {
    "rename": (
        "Update name and description",
        {
            "name": "helpline-assistant-v2",
            "description": "Answers beneficiary questions in English and Hindi",
        },
    ),
}

CONFIG_VERSION_CREATE_EXAMPLES: Examples = {
    "change_params": (
        "Change model and temperature",
        {
            "config_blob": {
                "completion": {"params": {"model": "gpt-4o", "temperature": 0.5}}
            },
            "commit_message": "Switch to gpt-4o with higher temperature",
        },
    ),
}
