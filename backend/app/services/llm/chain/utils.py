"""Utility functions for LLM chain operations, including speech-to-speech helpers."""

from typing import get_args

from app.models.llm.constants import STSLanguageCode

# BCP-47 language codes accepted by the speech-to-speech endpoint. Derived from
# STSLanguageCode (app/models/llm/constants.py) so the request model and this
# set never drift apart.
SUPPORTED_LANGUAGE_CODES: set[str] = set(get_args(STSLanguageCode))

DEFAULT_RAG_INSTRUCTIONS = (
    "Answer the user's question using the provided knowledge base. "
    "Be concise and accurate."
)
