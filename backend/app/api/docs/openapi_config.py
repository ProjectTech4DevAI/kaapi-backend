"""
OpenAPI schema customization for ReDoc documentation.

This module contains tag metadata and custom OpenAPI extensions
for organizing and enhancing the API documentation.
"""

# Tag metadata for organizing endpoints in documentation
tags_metadata = [
    {
        "name": "Onboarding",
        "description": "Getting started with the platform",
    },
    {
        "name": "Documents",
        "description": "Document upload, transformation, and management operations",
    },
    {
        "name": "Collections",
        "description": "Collection creation, deletion, and management for vector stores and assistants",
    },
    {
        "name": "Config Management",
        "description": "Configuration management operations",
    },
    {
        "name": "LLM",
        "description": "Large Language Model inference and interaction endpoints",
    },
    {
        "name": "Evaluation",
        "description": "Dataset upload, running evaluations, listing datasets as well as evaluations",
    },
    {
        "name": "Fine Tuning",
        "description": "Fine tuning LLM for specific use cases by providing labelled dataset",
    },
    {
        "name": "Model Evaluation",
        "description": "Fine tuned model performance evaluation and benchmarking",
    },
    {
        "name": "Responses",
        "description": "OpenAI Responses API integration for managing LLM conversations",
    },
    {
        "name": "OpenAI Conversations",
        "description": "OpenAI conversation management and interaction",
    },
    {
        "name": "Users",
        "description": "User account management and operations",
    },
    {
        "name": "Organizations",
        "description": "Organization management and settings",
    },
    {
        "name": "Projects",
        "description": "Project management operations",
    },
    {
        "name": "API Keys",
        "description": "API key generation and management",
    },
    {
        "name": "Credentials",
        "description": "Credential management and authentication",
    },
    {"name": "Login", "description": "User authentication and login operations"},
    {
        "name": "Assistants",
        "description": "[**Deprecated**] OpenAI Assistant creation and management. This feature will be removed in a future version.",
    },
    {
        "name": "Threads",
        "description": "[**Deprecated**] Conversation thread management for assistants. This feature will be removed in a future version.",
    },
]

# ReDoc-specific extension: x-tagGroups for hierarchical organization
# This creates collapsible groups in the ReDoc sidebar
tag_groups = [
    {"name": "Get Started", "tags": ["Onboarding"]},
    {
        "name": "Capabilities",
        "tags": [
            "Documents",
            "Collections",
            "Config Management",
            "LLM",
            "Evaluation",
            "Fine Tuning",
            "Model Evaluation",
            "Responses",
            "OpenAI Conversations",
            "Assistants",
            "Threads",
        ],
    },
    {
        "name": "Administration",
        "tags": [
            "Users",
            "Organizations",
            "Projects",
            "API Keys",
            "Credentials",
            "Login",
        ],
    },
]


def customize_openapi_schema(openapi_schema: dict) -> dict:
    """
    Add custom OpenAPI extensions to the schema.

    Args:
        openapi_schema: The base OpenAPI schema from FastAPI

    Returns:
        The customized OpenAPI schema with x-tagGroups and other extensions
    """
    openapi_schema["x-tagGroups"] = tag_groups
    deprecated_tags = ["Assistants", "Threads"]

    for _, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                operation_tags = operation.get("tags", [])
                if any(tag in deprecated_tags for tag in operation_tags):
                    operation["x-badges"] = [{"name": "Deprecated", "color": "orange"}]

    return openapi_schema
