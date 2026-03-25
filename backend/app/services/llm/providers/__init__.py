from app.services.llm.providers.base import BaseProvider
from app.services.llm.providers.registry import (
    LLMProvider,
    get_llm_provider,
)


def __getattr__(name: str):
    if name == "OpenAIProvider":
        from app.services.llm.providers.oai import OpenAIProvider
        return OpenAIProvider
    if name == "GoogleAIProvider":
        from app.services.llm.providers.gai import GoogleAIProvider
        return GoogleAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
