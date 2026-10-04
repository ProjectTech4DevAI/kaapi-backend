import asyncio
import json
from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import settings
from app.services.agent import tools as agent_tools
from app.services.agent.executor import _build_request_target, execute_tool_call
from app.services.agent.tools import AgentTool

Handler = Callable[[httpx.Request], httpx.Response]

CALLER_HEADERS = {
    "X-API-KEY": "ApiKey caller-key",
    "Authorization": "Bearer caller-token",
    "X-Request-ID": "req-123",
}


class RecordingTransport(httpx.MockTransport):
    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(record)


def _json_handler(body: object, status_code: int = 200) -> Handler:
    return lambda request: httpx.Response(status_code, json=body)


async def _run(
    handler: Handler,
    name: str,
    raw_args: dict[str, object],
    forwarded_headers: dict[str, str] | None = None,
):
    transport = RecordingTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://kaapi-internal"
    ) as http_client:
        outcome = await execute_tool_call(
            http_client=http_client,
            forwarded_headers=(
                CALLER_HEADERS if forwarded_headers is None else forwarded_headers
            ),
            name=name,
            raw_args=raw_args,
        )
    return outcome, transport.requests


class _SlugArgs(BaseModel):
    slug: str


def _slug_tool() -> AgentTool:
    return AgentTool(
        name="get_thing",
        description="test tool with a free-text path param",
        method="GET",
        path=f"{settings.API_V1_STR}/things/{{slug}}",
        args_model=_SlugArgs,
    )


@pytest.mark.asyncio
class TestRejectedBeforeSending:
    async def test_unknown_tool_is_error_and_sends_nothing(self) -> None:
        outcome, requests = await _run(_json_handler({}), "delete_everything", {})

        assert outcome.is_error is True
        assert outcome.status_code is None
        assert "Unknown tool 'delete_everything'" in outcome.content
        assert "list_evaluation_runs" in outcome.content
        assert requests == []

    @pytest.mark.parametrize(
        "raw_args, expected_fragment",
        [
            ({"evaluation_id": 1, "get_trace_info": True}, "get_trace_info"),
            ({"evaluation_id": "not-a-number"}, "evaluation_id"),
            ({}, "evaluation_id"),
            ({"evaluation_id": 1, "headers": {"X-API-KEY": "x"}}, "headers"),
        ],
        ids=["extra_arg", "wrong_type", "missing", "header_smuggling"],
    )
    async def test_invalid_args_are_rejected(
        self, raw_args: dict[str, object], expected_fragment: str
    ) -> None:
        outcome, requests = await _run(
            _json_handler({}), "get_evaluation_run", raw_args
        )

        assert outcome.is_error is True
        assert outcome.status_code is None
        assert outcome.content.startswith("Invalid arguments")
        assert expected_fragment in outcome.content
        assert outcome.arguments == raw_args
        assert requests == []

    @pytest.mark.parametrize(
        "traversal", ["../users", "1/../../users", "1%2F..%2Fusers"]
    )
    async def test_path_traversal_in_typed_param_is_rejected(
        self, traversal: str
    ) -> None:
        outcome, requests = await _run(
            _json_handler({}), "get_evaluation_run", {"evaluation_id": traversal}
        )

        assert outcome.is_error is True
        assert requests == []

    async def test_non_get_tool_is_refused_at_send_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = _slug_tool()
        # Simulates a registry entry mutated after the __post_init__ check.
        object.__setattr__(tool, "method", "DELETE")
        monkeypatch.setitem(agent_tools.TOOLS_BY_NAME, tool.name, tool)
        monkeypatch.setattr(
            "app.services.agent.executor.TOOLS_BY_NAME", agent_tools.TOOLS_BY_NAME
        )

        outcome, requests = await _run(_json_handler({}), tool.name, {"slug": "a"})

        assert outcome.is_error is True
        assert "not permitted" in outcome.content
        assert requests == []


class TestBuildRequestTarget:
    @pytest.mark.parametrize(
        "slug, expected_segment",
        [
            ("../../users/me", "..%2F..%2Fusers%2Fme"),
            ("a/b", "a%2Fb"),
            ("x?admin=1#frag", "x%3Fadmin%3D1%23frag"),
        ],
    )
    def test_str_path_values_cannot_add_segments(
        self, slug: str, expected_segment: str
    ) -> None:
        tool = _slug_tool()

        path, query = _build_request_target(tool, _SlugArgs(slug=slug))

        assert path == f"{settings.API_V1_STR}/things/{expected_segment}"
        assert query == {}

    def test_path_params_excluded_from_query(self) -> None:
        tool = agent_tools.TOOLS_BY_NAME["get_collection"]
        collection_id = uuid4()

        path, query = _build_request_target(
            tool, tool.args_model.model_validate({"collection_id": str(collection_id)})
        )

        assert path == f"{settings.API_V1_STR}/collections/{collection_id}"
        assert query == {"include_docs": True, "limit": 50}


@pytest.mark.asyncio
class TestRequestSent:
    async def test_none_query_params_are_omitted(self) -> None:
        outcome, [request] = await _run(
            _json_handler({"success": True, "data": []}), "list_evaluation_runs", {}
        )

        assert outcome.is_error is False
        assert request.method == "GET"
        assert request.url.path == f"{settings.API_V1_STR}/evaluations"
        assert dict(request.url.params) == {"limit": "10", "offset": "0"}

    async def test_only_credential_and_request_id_headers_forwarded(self) -> None:
        forwarded = {
            **CALLER_HEADERS,
            "Cookie": "access_token=leak",
            "X-Forwarded-For": "10.0.0.1",
            "Host": "evil.example",
        }

        _, [request] = await _run(
            _json_handler({"success": True, "data": []}),
            "list_evaluation_runs",
            {},
            forwarded_headers=forwarded,
        )

        assert request.headers["X-API-KEY"] == "ApiKey caller-key"
        assert request.headers["Authorization"] == "Bearer caller-token"
        assert request.headers["X-Request-ID"] == "req-123"
        assert "cookie" not in request.headers
        assert "x-forwarded-for" not in request.headers
        assert request.headers["host"] == "kaapi-internal"


@pytest.mark.asyncio
class TestResponseHandling:
    @pytest.mark.parametrize("status_code", [403, 404, 422, 500, 503])
    async def test_error_status_is_error_with_detail(self, status_code: int) -> None:
        outcome, _ = await _run(
            _json_handler(
                {"success": False, "data": None, "error": "Evaluation run not found"},
                status_code=status_code,
            ),
            "get_evaluation_run",
            {"evaluation_id": 7},
        )

        assert outcome.is_error is True
        assert outcome.status_code == status_code
        assert json.loads(outcome.content) == {
            "status_code": status_code,
            "error": "Evaluation run not found",
        }

    async def test_fastapi_detail_body_is_surfaced(self) -> None:
        outcome, _ = await _run(
            _json_handler({"detail": "Not Found"}, status_code=404),
            "get_evaluation_run",
            {"evaluation_id": 7},
        )

        assert json.loads(outcome.content)["error"] == "Not Found"

    async def test_plain_text_error_body_is_surfaced(self) -> None:
        outcome, _ = await _run(
            lambda request: httpx.Response(500, text="Internal Server Error"),
            "get_evaluation_run",
            {"evaluation_id": 7},
        )

        assert outcome.is_error is True
        assert json.loads(outcome.content) == {
            "status_code": 500,
            "error": "Internal Server Error",
        }

    async def test_redirect_is_not_followed(self) -> None:
        outcome, requests = await _run(
            lambda request: httpx.Response(
                307, headers={"Location": "https://evil.example/steal"}
            ),
            "get_evaluation_run",
            {"evaluation_id": 7},
        )

        assert outcome.is_error is True
        assert outcome.status_code == 307
        assert len(requests) == 1

    async def test_non_json_success_body_is_error(self) -> None:
        outcome, _ = await _run(
            lambda request: httpx.Response(200, text="<html>hi</html>"),
            "list_collections",
            {},
        )

        assert outcome.is_error is True
        assert outcome.status_code == 200
        assert "non-JSON" in outcome.content

    async def test_envelope_unwrapped_to_data(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": True,
                    "data": [{"id": "c1"}],
                    "error": None,
                    "metadata": None,
                }
            ),
            "list_collections",
            {},
        )

        assert outcome.is_error is False
        assert json.loads(outcome.content) == [{"id": "c1"}]

    async def test_envelope_metadata_kept(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": True,
                    "data": [{"id": "d1", "fname": "a.pdf"}],
                    "metadata": {"has_more": True},
                }
            ),
            "list_documents",
            {},
        )

        assert json.loads(outcome.content) == {
            "data": [{"id": "d1", "fname": "a.pdf"}],
            "metadata": {"has_more": True},
        }

    async def test_success_false_on_200_keeps_error_reason(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {"success": False, "data": {"id": 7}, "error": "Scores still pending"}
            ),
            "get_evaluation_run",
            {"evaluation_id": 7},
        )

        assert outcome.is_error is False
        assert json.loads(outcome.content) == {
            "data": {"id": 7},
            "error": "Scores still pending",
        }

    async def test_long_result_is_truncated_with_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "AGENT_TOOL_RESULT_MAX_CHARS", 20)

        outcome, _ = await _run(
            _json_handler({"success": True, "data": "x" * 100}),
            "list_collections",
            {},
        )

        # json.dumps("x" * 100) is 102 chars: 20 kept, 82 omitted.
        assert outcome.content == (
            '"'
            + "x" * 19
            + "...[truncated 82 chars; narrow the query with limit/offset]"
        )

    async def test_short_result_is_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "AGENT_TOOL_RESULT_MAX_CHARS", 102)

        outcome, _ = await _run(
            _json_handler({"success": True, "data": "x" * 100}),
            "list_collections",
            {},
        )

        assert outcome.content == '"' + "x" * 100 + '"'


@pytest.mark.asyncio
class TestTransportFailures:
    async def test_timeout_is_error(self) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out", request=request)

        outcome, _ = await _run(slow, "get_evaluation_run", {"evaluation_id": 7})

        assert outcome.is_error is True
        assert outcome.status_code is None
        assert "timed out" in outcome.content
        assert outcome.arguments == {"evaluation_id": 7}

    async def test_connection_error_is_error(self) -> None:
        def broken(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        outcome, _ = await _run(broken, "list_collections", {})

        assert outcome.is_error is True
        assert outcome.status_code is None
        assert "ConnectError" in outcome.content

    async def test_slow_in_process_route_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "AGENT_TOOL_TIMEOUT_SECONDS", 0.1)
        slow_app = FastAPI()

        @slow_app.get(f"{settings.API_V1_STR}/collections")
        async def slow_collections() -> dict[str, object]:
            await asyncio.sleep(1)
            return {"success": True, "data": []}

        # The production transport, unlike MockTransport above.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=slow_app),
            base_url="http://kaapi-internal",
        ) as http_client:
            outcome = await execute_tool_call(
                http_client=http_client,
                forwarded_headers={},
                name="list_collections",
                raw_args={},
            )

        assert outcome.is_error is True
        assert "timed out" in outcome.content
