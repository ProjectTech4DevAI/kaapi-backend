"""Executes one model-requested tool call against the real Kaapi GET route, in-process.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from app.core.config import settings
from app.core.security import API_KEY_HEADER_NAME
from app.services.agent.tools import (
    ALLOWED_TOOL_METHODS,
    TOOLS_BY_NAME,
    AgentTool,
    QueryParamValue,
)

logger = logging.getLogger(__name__)

AUTHORIZATION_HEADER = "Authorization"
REQUEST_ID_HEADER = "X-Request-ID"
FORWARDABLE_HEADERS = frozenset(
    {API_KEY_HEADER_NAME, AUTHORIZATION_HEADER, REQUEST_ID_HEADER}
)

_ENVELOPE_DATA_KEY = "data"
_ENVELOPE_METADATA_KEY = "metadata"
_ENVELOPE_SUCCESS_KEY = "success"
_ERROR_BODY_KEYS = ("error", "errors", "detail")
_HAS_MORE_KEY = "has_more"
# One extra row tells whether a next page exists, since routes report it inconsistently.
_PAGE_PROBE_ROWS = 1
_TRUNCATION_HINT = "narrow the query with limit/offset"


@dataclass(frozen=True)
class ToolCallOutcome:
    content: str
    is_error: bool
    status_code: int | None
    duration_ms: int
    arguments: dict[str, Any]


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _format_validation_error(exc: ValidationError) -> str:
    problems: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        problems.append(f"{location}: {error['msg']}")
    return "Invalid arguments — " + "; ".join(problems)


def _requested_page_size(tool: AgentTool, arguments: dict[str, Any]) -> int | None:
    if tool.page_size_param is None:
        return None
    page_size = arguments.get(tool.page_size_param)
    return page_size if isinstance(page_size, int) else None


def _build_request_target(
    tool: AgentTool, validated_args: BaseModel
) -> tuple[str, dict[str, QueryParamValue]]:
    dumped = validated_args.model_dump(mode="json", exclude_none=True)
    page_size = _requested_page_size(tool, dumped)

    path_values: dict[str, str] = {}
    for param_name in tool.path_param_names:
        # safe="" also encodes "/", so a value can never add path segments.
        path_values[param_name] = quote(str(dumped[param_name]), safe="")
    path = tool.path.format(**path_values)

    query_params: dict[str, QueryParamValue] = {}
    for key, value in dumped.items():
        if key not in tool.path_param_names:
            query_params[key] = value
    if tool.page_size_param is not None and page_size is not None:
        query_params[tool.page_size_param] = page_size + _PAGE_PROBE_ROWS
    # Applied last so a fixed param always wins over anything model-derived.
    for key, value in tool.fixed_query_params.items():
        query_params[key] = value
    return path, query_params


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase

    if isinstance(body, dict):
        for key in _ERROR_BODY_KEYS:
            detail = body.get(key)
            if detail:
                return detail if isinstance(detail, str) else json.dumps(detail)
    return json.dumps(body, default=str, ensure_ascii=False)


def _unwrap_envelope(
    tool: AgentTool, body: JsonValue, page_size: int | None
) -> JsonValue:
    if not isinstance(body, dict) or _ENVELOPE_DATA_KEY not in body:
        return tool.projector(body) if tool.projector else body

    data = body[_ENVELOPE_DATA_KEY]
    metadata: dict[str, JsonValue] = {}
    if page_size is not None and isinstance(data, list):
        metadata[_HAS_MORE_KEY] = len(data) > page_size
        data = data[:page_size]
    if tool.projector:
        data = tool.projector(data)

    # Some routes return 200 with success=false (e.g. partial score fetch); keep the reason.
    error = body.get("error") if body.get(_ENVELOPE_SUCCESS_KEY) is False else None
    if not metadata and not error:
        return data

    wrapped: dict[str, JsonValue] = {_ENVELOPE_DATA_KEY: data}
    if metadata:
        wrapped[_ENVELOPE_METADATA_KEY] = metadata
    if error:
        wrapped["error"] = error
    return wrapped


def _truncate(content: str) -> str:
    max_chars = settings.AGENT_TOOL_RESULT_MAX_CHARS
    if len(content) <= max_chars:
        return content
    omitted = len(content) - max_chars
    return f"{content[:max_chars]}...[truncated {omitted} chars; {_TRUNCATION_HINT}]"


async def execute_tool_call(
    *,
    http_client: httpx.AsyncClient,
    forwarded_headers: dict[str, str],
    name: str,
    raw_args: dict[str, Any],
) -> ToolCallOutcome:
    started_at = time.perf_counter()

    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return ToolCallOutcome(
            content=(
                f"Unknown tool '{name}'. Available tools: "
                f"{', '.join(sorted(TOOLS_BY_NAME))}."
            ),
            is_error=True,
            status_code=None,
            duration_ms=_elapsed_ms(started_at),
            arguments=raw_args,
        )

    try:
        validated_args = tool.args_model.model_validate(raw_args)
    except ValidationError as exc:
        message = _format_validation_error(exc)
        return ToolCallOutcome(
            content=message,
            is_error=True,
            status_code=None,
            duration_ms=_elapsed_ms(started_at),
            arguments=raw_args,
        )

    arguments = validated_args.model_dump(mode="json", exclude_none=True)
    page_size = _requested_page_size(tool, arguments)
    path, query_params = _build_request_target(tool, validated_args)

    # Re-checked at send time so a registry mutation can't smuggle in a write.
    if tool.method not in ALLOWED_TOOL_METHODS:
        return ToolCallOutcome(
            content=f"Tool '{name}' is not permitted (read-only agent).",
            is_error=True,
            status_code=None,
            duration_ms=_elapsed_ms(started_at),
            arguments=arguments,
        )

    headers: dict[str, str] = {}
    for header_name, header_value in forwarded_headers.items():
        if header_name in FORWARDABLE_HEADERS:
            headers[header_name] = header_value

    try:
        # ASGITransport ignores httpx timeouts, so the in-process call is bounded here.
        response = await asyncio.wait_for(
            http_client.request(
                tool.method,
                path,
                params=query_params,
                headers=headers,
                follow_redirects=False,
            ),
            timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
        )
    except (TimeoutError, httpx.TimeoutException) as exc:
        # Must come before RequestError — TimeoutException is a subclass.
        logger.error(
            f"[execute_tool_call] [KAAPI] Internal tool request timed out "
            f"(code: {type(exc).__name__}) | tool: {name}, "
            f"duration_ms: {_elapsed_ms(started_at)}",
            exc_info=True,
        )
        return ToolCallOutcome(
            content=(
                f"[KAAPI] The request timed out (code: {type(exc).__name__}). "
                "Retry with a smaller limit."
            ),
            is_error=True,
            status_code=None,
            duration_ms=_elapsed_ms(started_at),
            arguments=arguments,
        )
    except httpx.RequestError as exc:
        logger.error(
            f"[execute_tool_call] [KAAPI] Internal tool request failed "
            f"(code: {type(exc).__name__}) | tool: {name}",
            exc_info=True,
        )
        return ToolCallOutcome(
            content=(
                f"[KAAPI] The request could not be completed "
                f"(code: {type(exc).__name__})."
            ),
            is_error=True,
            status_code=None,
            duration_ms=_elapsed_ms(started_at),
            arguments=arguments,
        )

    status_code = response.status_code
    duration_ms = _elapsed_ms(started_at)

    # Redirects are not followed, so any non-2xx is a failed lookup.
    if not response.is_success:
        detail = _extract_error_detail(response)
        # 5xx is Kaapi breaking (alert-worthy); 4xx is a bad lookup by the model.
        log = logger.error if status_code >= 500 else logger.warning
        log(
            f"[execute_tool_call] Tool call failed | tool: {name}, "
            f"status: {status_code}, duration_ms: {duration_ms}"
        )
        return ToolCallOutcome(
            content=_truncate(
                json.dumps(
                    {"status_code": status_code, "error": detail},
                    default=str,
                    ensure_ascii=False,
                )
            ),
            is_error=True,
            status_code=status_code,
            duration_ms=duration_ms,
            arguments=arguments,
        )

    try:
        body = response.json()
    except ValueError:
        logger.warning(
            f"[execute_tool_call] [KAAPI] Tool response was not JSON | tool: {name}, "
            f"status: {status_code}"
        )
        return ToolCallOutcome(
            content="[KAAPI] The endpoint returned a non-JSON response.",
            is_error=True,
            status_code=status_code,
            duration_ms=duration_ms,
            arguments=arguments,
        )

    payload = _unwrap_envelope(tool, body, page_size)
    content = _truncate(json.dumps(payload, default=str, ensure_ascii=False))

    return ToolCallOutcome(
        content=content,
        is_error=False,
        status_code=status_code,
        duration_ms=duration_ms,
        arguments=arguments,
    )
