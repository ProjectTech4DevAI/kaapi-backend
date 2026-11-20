"""Request-body examples for the Threads routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

THREAD_CREATE_EXAMPLES: Examples = {
    "new_thread": (
        "Ask a question on a new thread",
        {
            "question": "What documents do I need to apply for the scholarship?",
            "assistant_id": "asst_abc123def456",
            "callback_url": "https://example.com/webhooks/threads",
            "remove_citation": True,
        },
    ),
}

THREAD_SYNC_EXAMPLES: Examples = {
    "continue_thread": (
        "Ask a follow-up question on an existing thread and wait for the answer",
        {
            "question": "What documents do I need to apply for the scholarship?",
            "assistant_id": "asst_abc123def456",
            "thread_id": "thread_abc123def456",
            "remove_citation": True,
        },
    ),
}

THREAD_START_EXAMPLES: Examples = {
    "basic": (
        "Start a thread and poll for the answer",
        {
            "question": "What documents do I need to apply for the scholarship?",
            "assistant_id": "asst_abc123def456",
            "remove_citation": True,
        },
    ),
}
