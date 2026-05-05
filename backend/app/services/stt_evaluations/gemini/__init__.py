"""Gemini integration — re-exports from app.core.batch for backwards compatibility."""

from app.core.batch.client import GeminiClient, GeminiClientError

__all__ = ["GeminiClient", "GeminiClientError"]
