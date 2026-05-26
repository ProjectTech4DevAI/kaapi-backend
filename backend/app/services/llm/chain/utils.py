"""Utility functions for LLM chain operations, including speech-to-speech helpers."""

# BCP-47 language codes accepted by the speech-to-speech endpoint.
SUPPORTED_LANGUAGE_CODES = {
    "auto",
    "unknown",
    # Primary Indian languages
    "en-IN",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    # Additional languages
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
}

DEFAULT_RAG_INSTRUCTIONS = (
    "Answer the user's question using the provided knowledge base. "
    "Be concise and accurate."
)
