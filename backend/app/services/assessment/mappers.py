import logging
import unicodedata

from google.genai import _transformers as genai_transformers
from sqlmodel import Session

from app.crud.model_config import is_reasoning_model

logger = logging.getLogger(__name__)


def normalize_llm_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace("\\r", "\r")
    text = text.replace('\\"', '"')
    text = text.replace("\\\\", "\\")

    text = unicodedata.normalize("NFC", text)

    return text


def _ensure_openai_strict_schema(schema: dict) -> dict:
    """Recursively add additionalProperties: false for OpenAI strict JSON schema validation."""
    normalized = dict(schema)

    if normalized.get("type") == "object":
        normalized["additionalProperties"] = False

    if "properties" in normalized:
        normalized["properties"] = {
            key: _ensure_openai_strict_schema(value)
            if isinstance(value, dict)
            else value
            for key, value in normalized["properties"].items()
        }

    items = normalized.get("items")
    if isinstance(items, dict):
        normalized["items"] = _ensure_openai_strict_schema(items)

    return normalized


def _strip_additional_properties(schema: dict) -> dict:
    """Recursively strip additionalProperties — unsupported by Google GenAI."""
    normalized_schema = dict(schema)
    normalized_schema.pop("additionalProperties", None)

    if "properties" in normalized_schema:
        normalized_schema["properties"] = {
            property_name: _strip_additional_properties(property_schema)
            if isinstance(property_schema, dict)
            else property_schema
            for property_name, property_schema in normalized_schema[
                "properties"
            ].items()
        }

    if "items" in normalized_schema and isinstance(normalized_schema["items"], dict):
        normalized_schema["items"] = _strip_additional_properties(
            normalized_schema["items"]
        )

    return normalized_schema


def _convert_json_schema_to_google(schema: dict) -> dict:
    """Convert a JSON Schema dict to Google GenAI's OpenAPI-style schema.

    Strips unsupported fields, then normalizes the schema through the Gemini SDK
    so enum/type values match Gemini's expected OpenAPI-flavored shape.
    """
    normalized_schema = _strip_additional_properties(schema)
    converted = genai_transformers.t_schema(None, normalized_schema)
    google_schema = (
        converted.model_dump(mode="json", exclude_none=True)
        if converted is not None
        else normalized_schema
    )

    if "properties" in google_schema and "propertyOrdering" not in google_schema:
        google_schema["propertyOrdering"] = list(
            normalized_schema.get("required", [])
        ) or list(google_schema["properties"].keys())

    return google_schema


def map_kaapi_to_openai_assessment_params(
    session: Session, kaapi_params: dict
) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to OpenAI batch assessment API parameters.

    Extends the base LLM mapper with structured output schema support via
    ``output_schema`` → ``text.format.json_schema`` (strict mode).

    Returns:
        Tuple of (OpenAI API params dict, list of warning strings)
    """
    openai_params: dict = {}
    warnings: list[str] = []

    model = kaapi_params.get("model")
    reasoning = kaapi_params.get("reasoning")
    effort = kaapi_params.get("effort") or reasoning
    summary = kaapi_params.get("summary")
    temperature = kaapi_params.get("temperature")
    top_p = kaapi_params.get("top_p")

    instructions = normalize_llm_text(kaapi_params.get("instructions"))
    knowledge_base_ids = kaapi_params.get("knowledge_base_ids")
    max_num_results = kaapi_params.get("max_num_results")
    response_format = kaapi_params.get("response_format")
    output_schema = kaapi_params.get("output_schema")

    support_reasoning = bool(model) and is_reasoning_model(
        session=session,
        provider="openai",
        model_name=model,
    )

    # max_output_tokens is intentionally omitted for batch assessment —
    # Indic feedback responses can be long and a stored token limit would truncate them.

    if support_reasoning:
        reasoning_payload: dict[str, object] = {}
        if effort is not None:
            reasoning_payload["effort"] = effort
        if summary is not None:
            reasoning_payload["summary"] = None if summary == "null" else summary
        if reasoning_payload:
            openai_params["reasoning"] = reasoning_payload
        if temperature is not None:
            warnings.append(
                "Parameter 'temperature' was suppressed because the selected model "
                "supports reasoning, and temperature is ignored when reasoning is enabled."
            )
        if top_p is not None:
            warnings.append(
                "Parameter 'top_p' was suppressed because the selected model "
                "supports reasoning, and top_p is ignored when reasoning is enabled."
            )
    else:
        if effort is not None or summary is not None:
            warnings.append(
                "Parameters 'effort'/'summary' were suppressed because the selected model "
                "does not support reasoning."
            )
        if temperature is not None:
            openai_params["temperature"] = temperature
        if top_p is not None:
            openai_params["top_p"] = top_p

    if model:
        openai_params["model"] = model

    if instructions:
        openai_params["instructions"] = instructions

    if output_schema is not None:
        openai_params["text"] = {
            "format": {
                "type": "json_schema",
                "name": "output",
                "strict": True,
                "schema": _ensure_openai_strict_schema(output_schema),
            }
        }
    elif response_format and response_format != "text":
        openai_params["text"] = {"format": {"type": response_format}}

    if knowledge_base_ids:
        openai_params["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": knowledge_base_ids,
                "max_num_results": max_num_results or 20,
            }
        ]

    return openai_params, warnings


def map_kaapi_to_google_assessment_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to Google AI (Gemini) API parameters.

    Returns:
        Tuple of (Google AI params dict, list of warning strings)
    """
    google_params: dict = {}
    warnings: list[str] = []

    model = kaapi_params.get("model")
    if not model:
        return {}, ["Missing required 'model' parameter"]

    google_params["model"] = model

    instructions = normalize_llm_text(kaapi_params.get("instructions"))
    if instructions:
        google_params["instructions"] = instructions

    temperature = kaapi_params.get("temperature")
    if temperature is not None:
        google_params["temperature"] = temperature

    top_p = kaapi_params.get("top_p")
    if top_p is not None:
        google_params["top_p"] = top_p

    max_output_tokens = kaapi_params.get("max_output_tokens")
    if max_output_tokens is not None:
        google_params["max_output_tokens"] = max_output_tokens

    thinking_level = kaapi_params.get("thinking_level")
    if thinking_level:
        google_params["thinking_config"] = {"thinking_level": thinking_level}

    reasoning = kaapi_params.get("reasoning")
    if reasoning:
        google_params["reasoning"] = reasoning

    output_schema = kaapi_params.get("output_schema")
    if output_schema is not None:
        google_params["output_schema"] = _convert_json_schema_to_google(output_schema)

    if kaapi_params.get("knowledge_base_ids"):
        warnings.append(
            "Parameter 'knowledge_base_ids' is not supported by Google AI and was ignored."
        )

    return google_params, warnings
