"""Request-body examples for the Responses routes.


"""

from app.api.openapi_docs import Examples

RESPONSE_CREATE_EXAMPLES: Examples = {
    "with_assistant": (
        "Ask a question via an assistant",
        {
            "assistant_id": "asst_abc123def456",
            "question": "What documents do I need to apply for the scholarship?",
            "callback_url": "https://example.com/webhooks/responses",
        },
    ),
}

RESPONSE_SYNC_EXAMPLES: Examples = {
    "inline_config": (
        "Ask a question with inline model settings",
        {
            "model": "gpt-4o-mini",
            "instructions": "Answer using only the provided documents.",
            "vector_store_ids": ["vs_abc123def456"],
            "max_num_results": 20,
            "temperature": 0.1,
            "question": "What documents do I need to apply for the scholarship?",
        },
    ),
}
