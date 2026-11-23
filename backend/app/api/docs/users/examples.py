"""Request-body examples for the Users routes.


"""

from app.api.openapi_docs import Examples

USER_SIGNUP_EXAMPLES: Examples = {
    "basic": (
        "Register a new user",
        {
            "email": "jane.doe@example.com",
            "password": "ChangeMe123!",
            "full_name": "Jane Doe",
        },
    ),
}

USER_UPDATE_ME_EXAMPLES: Examples = {
    "basic": (
        "Update name and email",
        {"full_name": "Jane Smith", "email": "jane.smith@example.com"},
    ),
}

USER_PASSWORD_UPDATE_EXAMPLES: Examples = {
    "basic": (
        "Change password",
        {"current_password": "ChangeMe123!", "new_password": "NewSecret456!"},
    ),
}
