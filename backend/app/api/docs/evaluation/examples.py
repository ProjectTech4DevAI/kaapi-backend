"""Request-body examples for the Evaluation routes (v1 and v2).


"""

from app.api.openapi_docs import Examples

EVALUATION_CREATE_EXAMPLES: Examples = {
    "fast": (
        "Start a fast evaluation run",
        {
            "dataset_id": 42,
            "experiment_name": "faq-bot-gpt-4o-baseline",
            "config_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "config_version": 1,
            "run_mode": "fast",
        },
    ),
}

EVALUATION_V2_CREATE_EXAMPLES: Examples = {
    "judged": (
        "Start a judged evaluation run",
        {
            "dataset_id": 42,
            "experiment_name": "faq-bot-judged-run",
            "config_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "config_version": 2,
            "duplication_factor": 3,
            "callback_url": "https://example.com/webhooks/kaapi",
        },
    ),
}

EVALUATION_IMPROVE_PROMPT_EXAMPLES: Examples = {
    "basic": (
        "Improve prompt with a webhook callback",
        {"callback_url": "https://example.com/webhooks/kaapi"},
    ),
}

EVALUATION_ITERATION_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Start a prompt-tuning iteration loop",
        {
            "dataset_id": 42,
            "experiment_name": "faq-bot-prompt-tuning",
            "config_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "config_version": 1,
            "max_rounds": 5,
            "callback_url": "https://example.com/webhooks/kaapi",
        },
    ),
}
