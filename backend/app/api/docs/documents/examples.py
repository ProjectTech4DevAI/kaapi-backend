"""Request-body examples for the Documents routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

DOCUMENT_UPLOAD_INITIATE_EXAMPLES: Examples = {
    "basic": ("Request an upload URL for a file", {"filename": "program-faq.pdf"}),
}
