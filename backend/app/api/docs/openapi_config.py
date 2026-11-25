"""
OpenAPI schema customization for ReDoc documentation.

This module contains tag metadata and custom OpenAPI extensions
for organizing and enhancing the API documentation.
"""

from app.api.docs.guardrails_schemas import GUARDRAILS_COMPONENT_SCHEMAS
from app.api.openapi_docs import (
    inject_component_schemas,
    register_component_schemas,
)

register_component_schemas(GUARDRAILS_COMPONENT_SCHEMAS)

TAG_DESCRIPTIONS: dict[str, str] = {
    "Onboarding": (
        "Getting started with the platform. One call creates (or reuses) an "
        "organization, a project, a user and an API key, and can save provider "
        "credentials for the project."
    ),
    "Documents": (
        "Upload, list, inspect and delete documents, and track document "
        "transformation jobs."
    ),
    "Documents v2": (
        "Direct-to-storage document upload: open an upload session, send the "
        "file straight to object storage, then register the document."
    ),
    "Collections": (
        "Collections are vector stores built from uploaded documents, used for "
        "file search / RAG. Create, update and delete them, and track their "
        "creation jobs."
    ),
    "Config Management": (
        "Versioned LLM configurations (provider, model, prompt, params). Every "
        "change creates a new version that other APIs reference by config id "
        "and version."
    ),
    "LLM": (
        "Run LLM calls with a stored or ad-hoc config: single calls, chains "
        "where each step feeds the next, and speech-to-speech. Results arrive "
        "by polling or webhook."
    ),
    "Chatbot": (
        "Build and run chatbots: create a bot, save and publish its versioned "
        "React Flow graph, then start sessions and exchange messages with end "
        "users."
    ),
    "Guardrails": (
        "Text safety validation: applying guardrails to text, and managing "
        "the validator configs, ban lists and LLM prompts they run on."
    ),
    "Evaluation": (
        "Evaluate LLM configs against golden Q&A datasets: upload datasets, "
        "start runs, read scores and improve prompts."
    ),
    "Evaluation v2": (
        "Kaapi-native evaluation: upload datasets, start runs, request prompt "
        "recommendations, and run a self-driving eval → improve-prompt → eval "
        "loop."
    ),
    "STT Evaluation": (
        "Speech-to-text evaluation: upload audio, build datasets, run "
        "transcription across models and record human feedback."
    ),
    "TTS Evaluation": (
        "Text-to-speech evaluation: build text datasets, run synthesis across "
        "models and record human feedback and scores."
    ),
    "Assessment": (
        "Run saved LLM configs over an uploaded dataset: manage datasets, start, "
        "retry and resume runs, export results and set post-processing."
    ),
    "Assessment (Experimental)": (
        "Experimental assessment APIs: submit an assessment against a saved "
        "config and poll or receive the result by webhook, and check whether a "
        "submission duplicates an entry in a corpus."
    ),
    "Fine Tuning": (
        "Start OpenAI fine-tuning jobs from a labelled CSV dataset and refresh "
        "their status."
    ),
    "Model Evaluation": (
        "Evaluate fine-tuned models for a document and find the best-performing one."
    ),
    "Responses": (
        "Call the OpenAI Responses API through Kaapi, synchronously or "
        "asynchronously with a webhook callback."
    ),
    "OpenAI Conversations": (
        "Read and delete stored OpenAI Responses conversations, looked up by id, "
        "response id or ancestor response id."
    ),
    "Assistants": (
        "[**Deprecated**] OpenAI Assistant creation and management. This "
        "feature will be removed in a future version."
    ),
    "Threads": (
        "[**Deprecated**] Conversation thread management for assistants. This "
        "feature will be removed in a future version."
    ),
    "Users": (
        "User accounts: read, update or delete your own profile and change your "
        "password; superusers can also create users."
    ),
    "Organizations": (
        "Create, read, update and delete organizations, the top-level tenant "
        "that owns projects."
    ),
    "Projects": (
        "Create, read, update and delete projects inside an organization, and "
        "manage project-level settings."
    ),
    "User Projects": (
        "Manage which users belong to a project: add by email, list and remove. "
        "Adding and removing are superuser only."
    ),
    "API Keys": (
        "Create, list, delete and verify the API keys a project uses for "
        "programmatic access (`X-API-KEY` header)."
    ),
    "Credentials": (
        "Store and manage provider credentials (OpenAI, Langfuse, etc.) for an "
        "organization and project, encrypted at rest."
    ),
    "Authentication": (
        "Login flows for the platform UI: Google OAuth, magic-link email login, "
        "invitation verification, project selection, token refresh and logout."
    ),
    "Login": "Username/password login that returns an access token.",
    "Features": (
        "Project-scoped feature flags that switch functionality on or off. "
        "Superuser only."
    ),
    "Model Config": (
        "Catalogue of supported provider models and their settings. Superusers "
        "create and update entries; everyone can list them."
    ),
    "Languages": (
        "Read-only list of the languages the platform supports, used by STT/TTS "
        "datasets and other language-aware APIs."
    ),
    "Analytics": (
        "Live monthly usage analytics for the current organization, as a table "
        "and as chart-ready series."
    ),
}

# Tag metadata for organizing endpoints in documentation
tags_metadata = [
    {"name": name, "description": description}
    for name, description in TAG_DESCRIPTIONS.items()
]

# ReDoc-specific extension: x-tagGroups for hierarchical organization
# This creates collapsible groups in the ReDoc sidebar
tag_groups = [
    {"name": "Get Started", "tags": ["Onboarding"]},
    {
        "name": "Capabilities",
        "tags": [
            "Documents",
            "Documents v2",
            "Collections",
            "Config Management",
            "LLM",
            "Chatbot",
            "Guardrails",
            "Evaluation",
            "Evaluation v2",
            "STT Evaluation",
            "TTS Evaluation",
            "Assessment",
            "Assessment (Experimental)",
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
            "User Projects",
            "API Keys",
            "Credentials",
            "Authentication",
            "Login",
            "Features",
            "Model Config",
            "Languages",
            "Analytics",
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
    openapi_schema = inject_component_schemas(openapi_schema)
    openapi_schema["x-tagGroups"] = tag_groups
    deprecated_tags = ["Assistants", "Threads"]

    for _, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                operation_tags = operation.get("tags", [])
                if any(tag in deprecated_tags for tag in operation_tags):
                    operation["x-badges"] = [{"name": "Deprecated", "color": "orange"}]

    return openapi_schema
