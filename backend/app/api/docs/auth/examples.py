"""Request-body examples for the Auth routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

AUTH_GOOGLE_EXAMPLES: Examples = {
    "basic": (
        "Sign in with a Google ID token",
        {
            "token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImV4YW1wbGUiLCJ0eXAiOiJKV1QifQ.your-google-id-token.signature"
        },
    ),
}

AUTH_SELECT_PROJECT_EXAMPLES: Examples = {
    "basic": ("Switch to a project", {"project_id": 1}),
}

AUTH_MAGIC_LINK_EXAMPLES: Examples = {
    "basic": ("Request a magic link", {"email": "jane.doe@example.com"}),
}
