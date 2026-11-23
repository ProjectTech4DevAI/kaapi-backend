"""Request-body examples for the Onboarding routes.


"""

from app.api.openapi_docs import Examples

ONBOARDING_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Onboard an organization, project and user",
        {
            "organization_name": "Acme Health Foundation",
            "project_name": "WhatsApp Helpline Bot",
            "email": "jane.doe@example.com",
            "password": "ChangeMe123!",
            "user_name": "Jane Doe",
            "credentials": [{"openai": {"api_key": "sk-proj-xxxxxxxx"}}],
        },
    ),
}
