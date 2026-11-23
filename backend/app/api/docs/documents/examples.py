"""Request-body examples for the Documents routes.


"""

from app.api.openapi_docs import Examples

DOCUMENT_UPLOAD_INITIATE_EXAMPLES: Examples = {
    "basic": ("Request an upload URL for a file", {"filename": "program-faq.pdf"}),
}
