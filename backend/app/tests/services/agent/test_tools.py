import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel

from app.core.config import settings
from app.main import app
from app.services.agent.tools import (
    ANTHROPIC_TOOL_DEFINITIONS,
    READ_ONLY_TOOLS,
    TOOLS_BY_NAME,
    AgentTool,
)

# Flags that make routes mint signed URLs or trigger recomputation.
FORBIDDEN_ARG_NAMES = {
    "include_url",
    "include_signed_url",
    "get_trace_info",
    "resync_score",
}


def _get_routes_by_path() -> dict[str, APIRoute]:
    return {
        route.path: route
        for route in app.routes
        if isinstance(route, APIRoute) and "GET" in route.methods
    }


class _IdArgs(BaseModel):
    item_id: int


class TestRegistryMatchesRealRoutes:
    @pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda t: t.name)
    def test_tool_targets_a_registered_get_route(self, tool: AgentTool) -> None:
        routes = _get_routes_by_path()

        assert tool.method == "GET"
        assert tool.path in routes, f"{tool.name}: no GET route at {tool.path}"

    @pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda t: t.name)
    def test_tool_query_args_are_real_route_query_params(self, tool: AgentTool) -> None:
        route = _get_routes_by_path()[tool.path]
        route_query_params = {param.alias for param in route.dependant.query_params}
        route_path_params = {param.alias for param in route.dependant.path_params}

        tool_query_args = set(tool.args_model.model_fields) - tool.path_param_names

        assert tool.path_param_names == route_path_params
        assert tool_query_args <= route_query_params, (
            f"{tool.name}: {sorted(tool_query_args - route_query_params)} "
            "are not query params of the route"
        )

    @pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda t: t.name)
    def test_tool_never_exposes_signed_url_or_recompute_flags(
        self, tool: AgentTool
    ) -> None:
        assert FORBIDDEN_ARG_NAMES.isdisjoint(tool.args_model.model_fields)

    @pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda t: t.name)
    def test_tool_args_reject_unknown_keys(self, tool: AgentTool) -> None:
        assert tool.args_model.model_json_schema().get("additionalProperties") is False

    def test_names_unique_and_definitions_cover_registry(self) -> None:
        assert len(TOOLS_BY_NAME) == len(READ_ONLY_TOOLS)
        assert [d["name"] for d in ANTHROPIC_TOOL_DEFINITIONS] == [
            t.name for t in READ_ONLY_TOOLS
        ]


class TestAgentToolDefinition:
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "get"])
    def test_non_get_method_rejected(self, method: str) -> None:
        with pytest.raises(ValueError, match="only \\['GET'\\] are allowed"):
            AgentTool(
                name="write_tool",
                description="x",
                method=method,
                path=f"{settings.API_V1_STR}/things/{{item_id}}",
                args_model=_IdArgs,
            )

    def test_path_param_missing_from_args_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="other_id"):
            AgentTool(
                name="bad_tool",
                description="x",
                method="GET",
                path=f"{settings.API_V1_STR}/things/{{item_id}}/{{other_id}}",
                args_model=_IdArgs,
            )

    def test_path_param_names_derived_from_template(self) -> None:
        tool = TOOLS_BY_NAME["get_config_version"]

        assert tool.path_param_names == {"config_id", "version_number"}

    def test_anthropic_definition_shape(self) -> None:
        definition = TOOLS_BY_NAME["get_evaluation_run"].to_anthropic_tool()

        assert definition["name"] == "get_evaluation_run"
        assert definition["description"]
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert schema["required"] == ["evaluation_id"]
        assert schema["properties"]["evaluation_id"]["type"] == "integer"


class TestEvaluationRunProjector:
    def _run(self) -> dict[str, object]:
        return {
            "id": 42,
            "run_name": "nightly",
            "status": "completed",
            "config_id": "c0ffee",
            "config_version": 2,
            "object_store_url": "s3://bucket/run.csv",
            "score_trace_url": "s3://bucket/traces.json",
            "score": {
                "summary_scores": [{"name": "accuracy", "avg": 0.9}],
                "traces": [{"trace_id": "t1", "question": "Q1"}],
            },
            "cost": {"total_cost_usd": 1.5},
        }

    def test_get_run_drops_traces_and_storage_urls(self) -> None:
        projector = TOOLS_BY_NAME["get_evaluation_run"].projector
        assert projector is not None

        assert projector(self._run()) == {
            "id": 42,
            "run_name": "nightly",
            "status": "completed",
            "config_id": "c0ffee",
            "config_version": 2,
            "score": {"summary_scores": [{"name": "accuracy", "avg": 0.9}]},
            "cost": {"total_cost_usd": 1.5},
        }

    def test_list_runs_projects_every_run(self) -> None:
        projector = TOOLS_BY_NAME["list_evaluation_runs"].projector
        assert projector is not None

        projected = projector([self._run(), {**self._run(), "id": 43, "score": None}])

        assert [r["id"] for r in projected] == [42, 43]
        assert projected[0]["score"] == {
            "summary_scores": [{"name": "accuracy", "avg": 0.9}]
        }
        assert projected[1]["score"] is None
        for run in projected:
            assert "object_store_url" not in run
            assert "score_trace_url" not in run

    def test_storage_urls_stripped_from_nested_documents(self) -> None:
        projector = TOOLS_BY_NAME["get_collection"].projector
        assert projector is not None

        projected = projector(
            {
                "id": "col-1",
                "documents": [
                    {"id": "d1", "fname": "a.pdf", "signed_url": "https://s3/signed"},
                    {"id": "d2", "fname": "b.pdf", "object_store_url": "s3://b.pdf"},
                ],
            }
        )

        assert projected == {
            "id": "col-1",
            "documents": [
                {"id": "d1", "fname": "a.pdf"},
                {"id": "d2", "fname": "b.pdf"},
            ],
        }
