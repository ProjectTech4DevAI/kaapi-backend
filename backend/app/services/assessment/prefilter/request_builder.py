"""Provider-aware batch request line builder for prefilter stages."""

from typing import Any

from app.core.config import settings
from app.services.assessment.mappers import _ensure_openai_strict_schema


def build_request_line(
    key: str,
    system: str,
    user_text: str,
    *,
    attachment_parts: list[dict[str, Any]] | None = None,
    response_schema: dict[str, Any] | None = None,
    file_search_store: str | None = None,
) -> dict[str, Any]:
    """Build one batch JSONL line shaped for the configured prefilter provider.

    ``attachment_parts`` are provider-shaped content parts (from the OpenAI/Gemini
    attachment resolvers) appended after the text part.
    """
    model = settings.ASSESSMENT_PREFILTER_MODEL

    if settings.ASSESSMENT_PREFILTER_PROVIDER == "openai":
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
        content.extend(attachment_parts or [])
        body: dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": [{"role": "user", "content": content}],
        }
        if response_schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "strict": True,
                    "schema": _ensure_openai_strict_schema(response_schema),
                }
            }
        if file_search_store:
            body["tools"] = [
                {
                    "type": "file_search",
                    "vector_store_ids": [file_search_store],
                    "max_num_results": 20,
                }
            ]
        return {
            "custom_id": key,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }

    parts: list[dict[str, Any]] = [{"text": user_text}]
    parts.extend(attachment_parts or [])
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [{"text": system}]},
        "model": f"models/{model}",
    }
    if response_schema is not None:
        request["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        }
    if file_search_store:
        request["tools"] = [
            {"fileSearch": {"fileSearchStoreNames": [file_search_store]}}
        ]
    return {"key": key, "request": request}
