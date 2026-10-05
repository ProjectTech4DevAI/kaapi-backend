import asyncio
import json
from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

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

    def test_fixed_param_wins_over_model_derived_query(self) -> None:
        # extra="allow" stands in for any future path that lets a model-derived
        # value collide with a fixed param (registry args models forbid extras).
        class _LooseArgs(BaseModel):
            model_config = ConfigDict(extra="allow")
            run_id: int

        tool = AgentTool(
            name="get_run",
            description="x",
            method="GET",
            path=f"{settings.API_V1_STR}/runs/{{run_id}}",
            args_model=_LooseArgs,
            fixed_query_params={"include_results": False},
        )

        _, query = _build_request_target(
            tool, _LooseArgs.model_validate({"run_id": 1, "include_results": True})
        )

        assert query == {"include_results": False}


@pytest.mark.asyncio
class TestRequestSent:
    async def test_none_query_params_are_omitted(self) -> None:
        outcome, [request] = await _run(
            _json_handler({"success": True, "data": []}), "list_evaluation_runs", {}
        )

        assert outcome.is_error is False
        assert request.method == "GET"
        assert request.url.path == f"{settings.API_V1_STR}/evaluations"
        # Default page size 10, plus the one-row has_more probe.
        assert dict(request.url.params) == {"limit": "11", "offset": "0"}

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

    async def test_paginated_envelope_carries_computed_has_more(self) -> None:
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

        # One row under the page size of 20: the route's has_more is overridden.
        assert json.loads(outcome.content) == {
            "data": [{"id": "d1", "fname": "a.pdf"}],
            "metadata": {"has_more": False},
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
            _json_handler({"success": True, "data": [{"description": "x" * 100}]}),
            "list_collections",
            {},
        )

        # '[{"description": "' is 18 chars + 100 x + '"}]' = 121: 20 kept, 101 omitted.
        assert outcome.content == (
            '[{"description": "'
            + "x" * 2
            + "...[truncated 101 chars; narrow the query with limit/offset]"
        )

    async def test_short_result_is_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "AGENT_TOOL_RESULT_MAX_CHARS", 121)

        outcome, _ = await _run(
            _json_handler({"success": True, "data": [{"description": "x" * 100}]}),
            "list_collections",
            {},
        )

        assert outcome.content == '[{"description": "' + "x" * 100 + '"}]'


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


@pytest.mark.asyncio
class TestFixedQueryParamsSent:
    @pytest.mark.parametrize(
        "name, raw_args, expected_path, expected_params",
        [
            (
                "get_stt_evaluation_run",
                {"run_id": 5},
                "/evaluations/stt/runs/5",
                {"include_results": "false"},
            ),
            (
                "get_tts_evaluation_run",
                {"run_id": 6},
                "/evaluations/tts/runs/6",
                {"include_results": "false"},
            ),
            (
                "get_stt_evaluation_dataset",
                {"dataset_id": 7},
                "/evaluations/stt/datasets/7",
                {"include_samples": "false"},
            ),
        ],
    )
    async def test_bulk_payload_switched_off_on_the_wire(
        self,
        name: str,
        raw_args: dict[str, object],
        expected_path: str,
        expected_params: dict[str, str],
    ) -> None:
        outcome, [request] = await _run(
            _json_handler({"success": True, "data": {"id": 1}}), name, raw_args
        )

        assert outcome.is_error is False
        assert request.url.path == f"{settings.API_V1_STR}{expected_path}"
        assert dict(request.url.params) == expected_params
        # The fixed param is an implementation detail, not something the model chose.
        assert outcome.arguments == raw_args

    @pytest.mark.parametrize(
        "name, raw_args",
        [
            ("get_stt_evaluation_run", {"run_id": 5, "include_results": True}),
            ("get_tts_evaluation_run", {"run_id": 6, "include_results": "true"}),
            ("get_stt_evaluation_dataset", {"dataset_id": 7, "include_samples": True}),
            ("get_stt_evaluation_run", {"run_id": 5, "result_limit": 1000}),
        ],
        ids=["stt_results", "tts_results", "stt_samples", "stt_result_limit"],
    )
    async def test_model_cannot_override_fixed_params(
        self, name: str, raw_args: dict[str, object]
    ) -> None:
        outcome, requests = await _run(_json_handler({}), name, raw_args)

        assert outcome.is_error is True
        assert outcome.content.startswith("Invalid arguments")
        assert requests == []


@pytest.mark.asyncio
class TestMetadataMinimization:
    async def test_non_pagination_metadata_dropped_and_bare_data_returned(
        self,
    ) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": True,
                    "data": {"id": 5, "run_name": "stt-nightly"},
                    "metadata": {
                        "results_total": 250,
                        "signed_url": "https://s3/signed",
                    },
                }
            ),
            "get_stt_evaluation_run",
            {"run_id": 5},
        )

        assert json.loads(outcome.content) == {"id": 5, "run_name": "stt-nightly"}

    async def test_only_pagination_keys_kept(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": True,
                    "data": [{"id": 5}],
                    "metadata": {
                        "total": 31,
                        "limit": 10,
                        "offset": 0,
                        "has_more": True,
                        "filters": {"status": "completed"},
                        "trace_id": "lf-trace-secret",
                    },
                }
            ),
            "list_stt_evaluation_runs",
            {},
        )

        assert json.loads(outcome.content) == {
            "data": [{"id": 5}],
            "metadata": {"has_more": False},
        }

    @pytest.mark.parametrize("metadata", [None, {}, "not-a-dict", ["total"]])
    async def test_route_metadata_shape_does_not_affect_paginated_result(
        self, metadata: object
    ) -> None:
        outcome, _ = await _run(
            _json_handler(
                {"success": True, "data": [{"id": "d1"}], "metadata": metadata}
            ),
            "list_documents",
            {},
        )

        assert json.loads(outcome.content) == {
            "data": [{"id": "d1"}],
            "metadata": {"has_more": False},
        }

    @pytest.mark.parametrize("metadata", [{}, "not-a-dict", ["total"]])
    async def test_empty_or_malformed_metadata_returns_bare_data(
        self, metadata: object
    ) -> None:
        outcome, _ = await _run(
            _json_handler(
                {"success": True, "data": [{"id": "c1"}], "metadata": metadata}
            ),
            "list_collections",
            {},
        )

        assert json.loads(outcome.content) == [{"id": "c1"}]

    async def test_success_false_error_kept_while_metadata_dropped(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": False,
                    "data": {"id": 7, "object_store_url": "s3://bucket/run.csv"},
                    "error": "Scores still pending",
                    "metadata": {"langfuse_trace": "lf-secret"},
                }
            ),
            "get_evaluation_run",
            {"evaluation_id": 7},
        )

        assert outcome.is_error is False
        assert json.loads(outcome.content) == {
            "data": {"id": 7},
            "error": "Scores still pending",
        }


PAGINATED_TOOL_DEFAULT_LIMITS = {
    "list_evaluation_runs": 10,
    "list_evaluation_datasets": 20,
    "list_documents": 20,
    "list_stt_evaluation_runs": 10,
    "list_tts_evaluation_runs": 10,
    "list_stt_evaluation_datasets": 20,
    "list_tts_evaluation_datasets": 20,
}

# Sits in the row's leading "id" so any leak of the probe row shows up even in a
# truncated prefix.
PROBE_ROW_ID = "probe-row-must-not-leak"


def _document_rows(count: int, *, with_probe: bool = False) -> list[dict[str, str]]:
    rows = [{"id": f"d{i}", "fname": f"doc-{i}.pdf"} for i in range(1, count + 1)]
    if with_probe:
        rows.append({"id": PROBE_ROW_ID, "fname": "probe.pdf"})
    return rows


def _page_envelope(rows: list[dict[str, str]], **extra: object) -> dict[str, object]:
    return {"success": True, "data": rows, **extra}


@pytest.mark.asyncio
class TestPaginationRequest:
    @pytest.mark.parametrize("name", PAGINATED_TOOL_DEFAULT_LIMITS)
    async def test_outgoing_limit_is_one_more_than_requested(self, name: str) -> None:
        outcome, [request] = await _run(
            _json_handler(_page_envelope([])), name, {"limit": 7}
        )

        assert outcome.is_error is False
        assert request.url.params["limit"] == "8"
        assert outcome.arguments["limit"] == 7

    @pytest.mark.parametrize("name, default", PAGINATED_TOOL_DEFAULT_LIMITS.items())
    async def test_default_page_size_also_gets_the_probe_row(
        self, name: str, default: int
    ) -> None:
        outcome, [request] = await _run(_json_handler(_page_envelope([])), name, {})

        assert request.url.params["limit"] == str(default + 1)
        assert outcome.arguments["limit"] == default

    async def test_non_paginated_tool_sends_limit_unchanged(self) -> None:
        collection_id = str(uuid4())

        outcome, [request] = await _run(
            _json_handler({"success": True, "data": {"id": collection_id}}),
            "get_collection",
            {"collection_id": collection_id, "limit": 30},
        )

        assert dict(request.url.params) == {"include_docs": "true", "limit": "30"}
        assert outcome.arguments == {
            "collection_id": collection_id,
            "include_docs": True,
            "limit": 30,
        }

    async def test_documents_page_sent_with_skip_and_arguments_keep_model_limit(
        self,
    ) -> None:
        outcome, [request] = await _run(
            _json_handler(_page_envelope([])),
            "list_documents",
            {"limit": 5, "skip": 15},
        )

        assert dict(request.url.params) == {"limit": "6", "skip": "15"}
        assert outcome.arguments == {"limit": 5, "skip": 15}


class TestPaginationRequestTarget:
    def test_fixed_param_still_wins_on_paginated_tool(self) -> None:
        # extra="allow" lets a model-derived key collide with the fixed param.
        class _LoosePageArgs(BaseModel):
            model_config = ConfigDict(extra="allow")
            limit: int = 10

        tool = AgentTool(
            name="list_runs",
            description="x",
            method="GET",
            path=f"{settings.API_V1_STR}/runs",
            args_model=_LoosePageArgs,
            fixed_query_params={"include_results": False},
            page_size_param="limit",
        )

        _, query = _build_request_target(
            tool,
            _LoosePageArgs.model_validate({"limit": 4, "include_results": True}),
        )

        assert query == {"limit": 5, "include_results": False}


@pytest.mark.asyncio
class TestPaginationResponse:
    async def test_probe_row_trimmed_and_has_more_true(self) -> None:
        outcome, _ = await _run(
            _json_handler(_page_envelope(_document_rows(3, with_probe=True))),
            "list_documents",
            {"limit": 3},
        )

        assert json.loads(outcome.content) == {
            "data": _document_rows(3),
            "metadata": {"has_more": True},
        }
        assert PROBE_ROW_ID not in outcome.content
        assert "probe.pdf" not in outcome.content

    @pytest.mark.parametrize("row_count", [0, 2, 3], ids=["empty", "short", "exact"])
    async def test_at_most_limit_rows_means_no_more(self, row_count: int) -> None:
        outcome, _ = await _run(
            _json_handler(_page_envelope(_document_rows(row_count))),
            "list_documents",
            {"limit": 3},
        )

        assert json.loads(outcome.content) == {
            "data": _document_rows(row_count),
            "metadata": {"has_more": False},
        }

    async def test_route_pagination_metadata_replaced_by_computed_has_more(
        self,
    ) -> None:
        outcome, _ = await _run(
            _json_handler(
                _page_envelope(
                    _document_rows(3),
                    metadata={"total": 99, "limit": 4, "offset": 0, "has_more": True},
                )
            ),
            "list_documents",
            {"limit": 3},
        )

        assert json.loads(outcome.content) == {
            "data": _document_rows(3),
            "metadata": {"has_more": False},
        }

    async def test_non_list_data_on_paginated_tool_has_no_metadata(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": True,
                    "data": {"id": "d1", "fname": "a.pdf"},
                    "metadata": {"has_more": True, "total": 1},
                }
            ),
            "list_documents",
            {},
        )

        assert json.loads(outcome.content) == {"id": "d1", "fname": "a.pdf"}

    async def test_non_paginated_list_is_bare_and_untrimmed(self) -> None:
        rows = [{"id": f"c{i}"} for i in range(1, 61)]

        outcome, _ = await _run(
            _json_handler(
                {"success": True, "data": rows, "metadata": {"has_more": True}}
            ),
            "list_collections",
            {},
        )

        assert json.loads(outcome.content) == rows

    async def test_success_false_error_kept_alongside_has_more(self) -> None:
        outcome, _ = await _run(
            _json_handler(
                {
                    "success": False,
                    "data": _document_rows(2, with_probe=True),
                    "error": "Some documents could not be loaded",
                    "metadata": {"total": 50},
                }
            ),
            "list_documents",
            {"limit": 2},
        )

        assert outcome.is_error is False
        assert json.loads(outcome.content) == {
            "data": _document_rows(2),
            "metadata": {"has_more": True},
            "error": "Some documents could not be loaded",
        }

    async def test_trimmed_page_that_fits_budget_is_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = {"data": _document_rows(2), "metadata": {"has_more": True}}
        # Exactly the trimmed page: an untrimmed page would overflow it.
        budget = len(json.dumps(expected))
        monkeypatch.setattr(settings, "AGENT_TOOL_RESULT_MAX_CHARS", budget)

        outcome, _ = await _run(
            _json_handler(_page_envelope(_document_rows(2, with_probe=True))),
            "list_documents",
            {"limit": 2},
        )

        assert json.loads(outcome.content) == expected

    async def test_probe_row_absent_when_truncation_kicks_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trimmed = json.dumps(
            {"data": _document_rows(2), "metadata": {"has_more": True}}
        )
        # One char short of the trimmed page; an untrimmed page's prefix of this
        # length would reach into the probe row's id.
        monkeypatch.setattr(settings, "AGENT_TOOL_RESULT_MAX_CHARS", len(trimmed) - 1)

        outcome, _ = await _run(
            _json_handler(_page_envelope(_document_rows(2, with_probe=True))),
            "list_documents",
            {"limit": 2},
        )

        assert outcome.content.startswith(trimmed[:-1])
        assert "...[truncated 1 chars;" in outcome.content
        assert "probe-row" not in outcome.content
