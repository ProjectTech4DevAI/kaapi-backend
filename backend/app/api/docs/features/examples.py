"""Request-body examples for the Feature Flags routes.


"""

from app.api.openapi_docs import Examples

FEATURE_FLAG_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Enable a flag for a project",
        {"key": "ASSESSMENT", "project_id": 1, "enabled": True},
    ),
}

FEATURE_FLAG_UPDATE_EXAMPLES: Examples = {
    "disable": (
        "Disable a flag for a project",
        {"key": "ASSESSMENT", "project_id": 1, "enabled": False},
    ),
}

FEATURE_FLAG_DELETE_EXAMPLES: Examples = {
    "basic": ("Remove a flag from a project", {"key": "ASSESSMENT", "project_id": 1}),
}
