"""Thin LLM client wrappers for OpenAI and Gemini. No Pydantic, no frameworks."""

from __future__ import annotations

import json
import base64
import logging
import httpx
from typing import Any

logger = logging.getLogger(__name__)


def _download_image_as_base64(url: str) -> tuple[str, str]:
    """Download image and return (base64_data, media_type)."""
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return base64.b64encode(resp.content).decode(), content_type


def call_openai(
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_urls: list[str] | None = None,
    output_schema: dict | None = None,
    temperature: float = 0.4,
) -> dict[str, Any]:
    """Call OpenAI chat completions and return parsed JSON dict."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    content: list[dict] = [{"type": "text", "text": user_text}]
    if image_urls:
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if output_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "strict": True, "schema": output_schema},
        }

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content
    return json.loads(raw)


def call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_urls: list[str] | None = None,
    output_schema: dict | None = None,
    temperature: float = 0.4,
) -> dict[str, Any]:
    """Call Google Gemini and return parsed JSON dict."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    parts: list[types.Part] = [types.Part.from_text(text=user_text)]

    if image_urls:
        for url in image_urls:
            try:
                b64_data, media_type = _download_image_as_base64(url)
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=media_type)
                )
            except Exception as e:
                logger.warning(f"[call_gemini] Failed to download image {url}: {e}")

    config: dict[str, Any] = {
        "temperature": temperature,
        "system_instruction": system_prompt,
    }

    if output_schema:
        # Strip additionalProperties for Gemini compatibility
        clean_schema = _strip_additional_properties(output_schema)
        config["response_mime_type"] = "application/json"
        config["response_schema"] = clean_schema

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(**config),
    )
    return json.loads(response.text)


def _strip_additional_properties(schema: dict) -> dict:
    schema = dict(schema)
    schema.pop("additionalProperties", None)
    if "properties" in schema:
        schema["properties"] = {
            k: _strip_additional_properties(v) if isinstance(v, dict) else v
            for k, v in schema["properties"].items()
        }
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _strip_additional_properties(schema["items"])
    return schema
