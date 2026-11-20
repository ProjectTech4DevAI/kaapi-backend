"""Request-body examples for the Assistants routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

ASSISTANT_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Create an assistant over a vector store",
        {
            "name": "Helpline Assistant",
            "instructions": "You are a helpful assistant. Answer questions using the program FAQ documents.",
            "model": "gpt-4o-mini",
            "vector_store_ids": ["vs_abc123def456"],
            "temperature": 0.1,
            "max_num_results": 20,
        },
    ),
}

ASSISTANT_UPDATE_EXAMPLES: Examples = {
    "add_vector_store": (
        "Update instructions and add a vector store",
        {
            "instructions": "You are a helpful assistant. Answer in the user's language using the FAQ documents.",
            "vector_store_ids_add": ["vs_xyz789ghi012"],
            "temperature": 0.2,
        },
    ),
}
