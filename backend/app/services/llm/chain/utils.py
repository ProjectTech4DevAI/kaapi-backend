"""Utility functions for LLM chain operations, including speech-to-speech helpers."""

from typing import get_args

from app.models.llm.constants import SUPPORTED_STS_LANGUAGE_CODES

SUPPORTED_LANGUAGE_CODES: set[str] = set(get_args(SUPPORTED_STS_LANGUAGE_CODES))

DEFAULT_RAG_INSTRUCTIONS = (
    "Answer the user's question using the provided knowledge base. "
    "Be concise and accurate."
)
