import re
from typing import Any

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.main import app
from app.services.agent.prompts import AGENT_SYSTEM_PROMPT
from app.services.agent.tools import (
    ANTHROPIC_TOOL_DEFINITIONS,
    READ_ONLY_TOOLS,
    TOOLS_BY_NAME,
    AgentTool,
    _require_projectors,
)

# Flags that make routes mint signed URLs or trigger recomputation.
FORBIDDEN_ARG_NAMES = {
    "include_url",
    "include_signed_url",
    "get_trace_info",
    "resync_score",
}

# Model-settable switches/pagers for per-row payloads. include_docs is the one
# include_* flag allowed: get_collection's document list is allowlisted.
BULK_PAYLOAD_ARG_PATTERN = re.compile(r"^(include_(?!docs$)|result_|sample_)")

SPEECH_ID_ONLY_TOOLS = {
    "get_stt_evaluation_run": "run_id",
    "get_tts_evaluation_run": "run_id",
    "get_stt_evaluation_dataset": "dataset_id",
}


def _get_routes_by_path() -> dict[str, APIRoute]:
    return {
        route.path: route
        for route in app.routes
        if isinstance(route, APIRoute) and "GET" in route.methods
    }


class _IdArgs(BaseModel):
    item_id: int


# name -> (offset param, default page size)
PAGINATED_TOOLS: dict[str, tuple[str, int]] = {
    "list_evaluation_runs": ("offset", 10),
    "list_evaluation_datasets": ("offset", 20),
    "list_documents": ("skip", 20),
    "list_stt_evaluation_runs": ("offset", 10),
    "list_tts_evaluation_runs": ("offset", 10),
    "list_stt_evaluation_datasets": ("offset", 20),
    "list_tts_evaluation_datasets": ("offset", 20),
}
MAX_PAGE_SIZE = 50
# The executor asks the route for one row past the page to compute has_more.
MAX_PROBE_LIMIT = MAX_PAGE_SIZE + 1


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
        fixed_params = set(tool.fixed_query_params)

        assert tool.path_param_names == route_path_params
        assert tool_query_args <= route_query_params, (
            f"{tool.name}: {sorted(tool_query_args - route_query_params)} "
            "are not query params of the route"
        )
        # A misspelled fixed param would be silently ignored by FastAPI and the
        # route's default-on bulk payload would come back.
        assert fixed_params <= route_query_params, (
            f"{tool.name}: fixed {sorted(fixed_params - route_query_params)} "
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

    @pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda t: t.name)
    def test_tool_never_exposes_bulk_payload_args(self, tool: AgentTool) -> None:
        exposed = [
            name
            for name in tool.args_model.model_fields
            if BULK_PAYLOAD_ARG_PATTERN.match(name)
        ]

        assert exposed == []

    @pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda t: t.name)
    def test_every_registry_tool_has_a_projector(self, tool: AgentTool) -> None:
        assert tool.projector is not None

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

    def test_page_size_param_missing_from_args_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="page_size_param 'page_size'"):
            AgentTool(
                name="bad_pager",
                description="x",
                method="GET",
                path=f"{settings.API_V1_STR}/things/{{item_id}}",
                args_model=_IdArgs,
                page_size_param="page_size",
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


class _PageArgs(BaseModel):
    limit: int = 10


def _project(tool_name: str, payload: Any) -> Any:
    projector = TOOLS_BY_NAME[tool_name].projector
    assert projector is not None
    return projector(payload)


class TestFixedQueryParams:
    def test_fixed_param_overlapping_args_field_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"fixed query params \['limit'\]"):
            AgentTool(
                name="bad_tool",
                description="x",
                method="GET",
                path=f"{settings.API_V1_STR}/things",
                args_model=_PageArgs,
                fixed_query_params={"limit": 5},
            )

    @pytest.mark.parametrize(
        "tool_name",
        [tool.name for tool in READ_ONLY_TOOLS if tool.fixed_query_params],
    )
    def test_registry_fixed_params_are_read_only(self, tool_name: str) -> None:
        fixed = TOOLS_BY_NAME[tool_name].fixed_query_params

        with pytest.raises(TypeError):
            fixed["include_results"] = True  # type: ignore[index]

    def test_fixed_params_detached_from_source_mapping(self) -> None:
        source = {"include_results": False}
        tool = AgentTool(
            name="pinned",
            description="x",
            method="GET",
            path=f"{settings.API_V1_STR}/things",
            args_model=_PageArgs,
            fixed_query_params=source,
        )

        source["include_results"] = True

        assert tool.fixed_query_params == {"include_results": False}

    @pytest.mark.parametrize(
        "tool_name, expected",
        [
            ("get_stt_evaluation_run", {"include_results": False}),
            ("get_tts_evaluation_run", {"include_results": False}),
            ("get_stt_evaluation_dataset", {"include_samples": False}),
        ],
    )
    def test_speech_get_tools_pin_bulk_payload_off(
        self, tool_name: str, expected: dict[str, bool]
    ) -> None:
        assert dict(TOOLS_BY_NAME[tool_name].fixed_query_params) == expected


class TestRequireProjectors:
    def test_tool_without_projector_raises(self) -> None:
        unprojected = AgentTool(
            name="raw_tool",
            description="x",
            method="GET",
            path=f"{settings.API_V1_STR}/things/{{item_id}}",
            args_model=_IdArgs,
        )

        with pytest.raises(ValueError, match="'raw_tool' has no projector"):
            _require_projectors((*READ_ONLY_TOOLS, unprojected))


class TestSpeechArgsAreIdOnly:
    @pytest.mark.parametrize("tool_name, id_field", SPEECH_ID_ONLY_TOOLS.items())
    def test_args_model_is_only_the_id(self, tool_name: str, id_field: str) -> None:
        assert set(TOOLS_BY_NAME[tool_name].args_model.model_fields) == {id_field}

    @pytest.mark.parametrize("tool_name, id_field", SPEECH_ID_ONLY_TOOLS.items())
    @pytest.mark.parametrize(
        "extra",
        [
            {"include_results": True},
            {"include_samples": True},
            {"result_limit": 1000},
            {"sample_offset": 5},
        ],
        ids=["include_results", "include_samples", "result_limit", "sample_offset"],
    )
    def test_extra_keys_rejected(
        self, tool_name: str, id_field: str, extra: dict[str, Any]
    ) -> None:
        args_model = TOOLS_BY_NAME[tool_name].args_model

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            args_model.model_validate({id_field: 1, **extra})


def _evaluation_run_payload() -> dict[str, Any]:
    return {
        "id": 42,
        "run_name": "nightly",
        "dataset_id": 3,
        "dataset_name": "faq",
        "config_id": "8f6c1c1e-5e4b-4c2a-9d3a-0b5f0c1d2e3f",
        "config_version": 2,
        "batch_job_id": 11,
        "embedding_batch_job_id": 12,
        "status": "completed",
        "run_mode": "batch",
        "object_store_url": "s3://bucket/run.csv",
        "score_trace_url": "s3://bucket/traces.json",
        "total_items": 30,
        "per_item_scores": {"trace-1": {"cosine": 0.4}},
        "ai_summary": "The model struggled with refund questions.",
        "unscoreable": {"count": 1},
        "is_score_updated": True,
        "is_judge_run": False,
        "error_message": None,
        "organization_id": 1,
        "project_id": 1,
        "inserted_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-02T00:00:00",
        "score": {
            "summary_scores": [
                {
                    "name": "cosine",
                    "avg": 0.81,
                    "std": 0.1,
                    "total_pairs": 30,
                    "data_type": "NUMERIC",
                    "distribution": {"0-0.5": 4, "0.5-1": 26},
                    "comment": "free text from the scorer",
                }
            ],
            "overall": {
                "overall_score": 0.8,
                "verdict": "pass",
                "breakdown": [
                    {"name": "Cosine", "key": "cosine", "score": 0.81, "weight": 1.0}
                ],
                "rationale": "LLM-written explanation",
            },
            "category_metrics": [
                {"category": "refunds", "total_evals": 10, "avg_cosine": 0.6}
            ],
            "traces": [
                {
                    "trace_id": "trace-1",
                    "question": "How do I get a refund?",
                    "llm_answer": "Contact support.",
                }
            ],
            "ai_summary": "Refund answers were weak.",
        },
        "cost": {
            "response": {
                "model": "gpt-4o",
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost_usd": 0.01,
            },
            "embedding": {"model": "text-embedding-3-large", "cost_usd": 0.001},
            "total_cost_usd": 0.011,
        },
    }


class TestEvaluationRunAllowlist:
    def test_keeps_run_aggregates_and_drops_row_level_and_storage_fields(
        self,
    ) -> None:
        assert _project("get_evaluation_run", _evaluation_run_payload()) == {
            "id": 42,
            "run_name": "nightly",
            "dataset_id": 3,
            "dataset_name": "faq",
            "config_id": "8f6c1c1e-5e4b-4c2a-9d3a-0b5f0c1d2e3f",
            "config_version": 2,
            "status": "completed",
            "run_mode": "batch",
            "total_items": 30,
            "is_judge_run": False,
            "error_message": None,
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-02T00:00:00",
            "score": {
                "summary_scores": [
                    {
                        "name": "cosine",
                        "avg": 0.81,
                        "std": 0.1,
                        "total_pairs": 30,
                        "data_type": "NUMERIC",
                        "distribution": {"0-0.5": 4, "0.5-1": 26},
                    }
                ],
                "overall": {
                    "overall_score": 0.8,
                    "verdict": "pass",
                    "breakdown": [
                        {
                            "name": "Cosine",
                            "key": "cosine",
                            "score": 0.81,
                            "weight": 1.0,
                        }
                    ],
                },
                "category_metrics": [
                    {"category": "refunds", "total_evals": 10, "avg_cosine": 0.6}
                ],
            },
            "cost": {
                "response": {
                    "model": "gpt-4o",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                    "cost_usd": 0.01,
                },
                "embedding": {"model": "text-embedding-3-large", "cost_usd": 0.001},
                "total_cost_usd": 0.011,
            },
        }

    def test_list_projects_each_run(self) -> None:
        projected = _project(
            "list_evaluation_runs",
            [_evaluation_run_payload(), {**_evaluation_run_payload(), "id": 43}],
        )

        assert [run["id"] for run in projected] == [42, 43]
        rendered = str(projected)
        for dropped in ("How do I get a refund?", "s3://", "trace-1", "struggled"):
            assert dropped not in rendered


class TestConfigVersionAllowlist:
    def test_keeps_routing_fields_and_drops_prompt_text(self) -> None:
        payload = {
            "id": "11111111-1111-1111-1111-111111111111",
            "config_id": "22222222-2222-2222-2222-222222222222",
            "version": 3,
            "commit_message": "tighten tone for refunds",
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "config_blob": {
                "completion": {
                    "type": "text",
                    "provider": "openai",
                    "params": {
                        "model": "gpt-4o",
                        "instructions": "You are a refunds bot. Never admit fault.",
                        "knowledge_base_ids": ["vs_abc", "vs_def"],
                        "temperature": 0.2,
                    },
                },
                "prompt_template": {"template": "Customer says: {{input}}"},
                "input_guardrails": [
                    {"validator_config_id": "33333333-3333-3333-3333-333333333333"}
                ],
                "output_guardrails": [],
            },
        }

        assert _project("get_config_version", payload) == {
            "config_id": "22222222-2222-2222-2222-222222222222",
            "version": 3,
            "inserted_at": "2026-01-01T00:00:00",
            "config_blob": {
                "completion": {
                    "type": "text",
                    "provider": "openai",
                    "params": {
                        "model": "gpt-4o",
                        "knowledge_base_ids": ["vs_abc", "vs_def"],
                    },
                },
            },
        }


def _speech_run_payload() -> dict[str, Any]:
    return {
        "id": 5,
        "run_name": "stt-nightly",
        "dataset_name": "hindi-calls",
        "type": "stt",
        "language_id": 2,
        "models": ["gemini-2.5-pro"],
        "dataset_id": 9,
        "status": "completed",
        "total_items": 2,
        "score": {
            "summary_scores": [{"name": "wer", "avg": 0.12}],
            "per_sample": [{"sample_id": 1, "wer": 0.5}],
        },
        "error_message": None,
        "run_metadata": {"prompt": "Transcribe the caller verbatim"},
        "organization_id": 1,
        "project_id": 1,
        "inserted_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "results": [
            {
                "id": 1,
                "transcription": "mera account band ho gaya",
                "ground_truth": "mera account band ho gaya hai",
                "signed_url": "https://s3/signed",
            }
        ],
        "results_total": 1,
    }


class TestSpeechRunAllowlist:
    @pytest.mark.parametrize(
        "tool_name",
        [
            "get_stt_evaluation_run",
            "get_tts_evaluation_run",
            "list_stt_evaluation_runs",
            "list_tts_evaluation_runs",
        ],
    )
    def test_drops_results_and_run_metadata(self, tool_name: str) -> None:
        assert _project(tool_name, _speech_run_payload()) == {
            "id": 5,
            "run_name": "stt-nightly",
            "dataset_id": 9,
            "dataset_name": "hindi-calls",
            "status": "completed",
            "total_items": 2,
            "language_id": 2,
            "models": ["gemini-2.5-pro"],
            "error_message": None,
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "score": {"summary_scores": [{"name": "wer", "avg": 0.12}]},
        }


class TestSpeechDatasetAllowlist:
    @pytest.mark.parametrize(
        "tool_name",
        [
            "get_stt_evaluation_dataset",
            "list_stt_evaluation_datasets",
            "get_tts_evaluation_dataset",
            "list_tts_evaluation_datasets",
        ],
    )
    def test_drops_samples_and_storage_urls(self, tool_name: str) -> None:
        payload = {
            "id": 9,
            "name": "hindi-calls",
            "description": "support calls",
            "type": "stt",
            "language_id": 2,
            "object_store_url": "s3://bucket/datasets/hindi.csv",
            "signed_url": "https://s3/signed",
            "dataset_metadata": {
                "sample_count": 2,
                "has_ground_truth_count": 1,
                "source_filename": "calls.csv",
            },
            "organization_id": 1,
            "project_id": 1,
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "samples": [
                {
                    "id": 1,
                    "ground_truth": "mera account band ho gaya",
                    "object_store_url": "s3://bucket/audio/1.mp3",
                }
            ],
        }

        assert _project(tool_name, payload) == {
            "id": 9,
            "name": "hindi-calls",
            "description": "support calls",
            "language_id": 2,
            "dataset_metadata": {"sample_count": 2, "has_ground_truth_count": 1},
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }


class TestEvaluationDatasetAllowlist:
    @pytest.mark.parametrize(
        "tool_name", ["get_evaluation_dataset", "list_evaluation_datasets"]
    )
    def test_keeps_counts_and_drops_storage_and_ids(self, tool_name: str) -> None:
        payload = {
            "dataset_id": 3,
            "dataset_name": "faq",
            "description": "golden FAQ",
            "total_items": 30,
            "original_items": 10,
            "duplication_factor": 3,
            "object_store_url": "s3://bucket/faq.csv",
            "langfuse_dataset_id": "lf-123",
            "signed_url": "https://s3/signed",
        }

        assert _project(tool_name, payload) == {
            "dataset_id": 3,
            "dataset_name": "faq",
            "description": "golden FAQ",
            "total_items": 30,
            "original_items": 10,
            "duplication_factor": 3,
        }


class TestCollectionAndDocumentAllowlist:
    def _collection(self) -> dict[str, Any]:
        return {
            "id": "c1",
            "name": "policies",
            "description": "HR policies",
            "knowledge_base_id": "vs_abc",
            "knowledge_base_provider": "openai",
            "llm_service_id": "asst_secret",
            "llm_service_name": "gpt-4o",
            "project_id": 1,
            "organization_id": 1,
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "deleted_at": None,
        }

    def test_list_collections_keeps_only_allowlisted_fields(self) -> None:
        collection_fields = {
            "id": "c1",
            "name": "policies",
            "description": "HR policies",
            "knowledge_base_id": "vs_abc",
            "knowledge_base_provider": "openai",
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }

        assert _project("list_collections", [self._collection()]) == [collection_fields]

    def test_get_collection_keeps_only_allowlisted_document_fields(self) -> None:
        payload = {
            **self._collection(),
            "documents": [
                {
                    "id": "d1",
                    "fname": "leave.pdf",
                    "object_store_url": "s3://bucket/leave.pdf",
                    "signed_url": "https://s3/signed",
                    "file_size": 2048,
                    "inserted_at": "2026-01-01T00:00:00",
                }
            ],
        }

        projected = _project("get_collection", payload)

        assert projected["documents"] == [
            {"id": "d1", "fname": "leave.pdf", "inserted_at": "2026-01-01T00:00:00"}
        ]
        assert "llm_service_id" not in projected
        assert "deleted_at" not in projected

    @pytest.mark.parametrize("tool_name", ["get_document", "list_documents"])
    def test_document_keeps_only_allowlisted_fields(self, tool_name: str) -> None:
        payload = {
            "id": "d1",
            "fname": "leave.pdf",
            "source_document_id": None,
            "object_store_url": "s3://bucket/leave.pdf",
            "signed_url": "https://s3/signed",
            "file_size": 2048,
            "project_id": 1,
            "is_deleted": False,
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }

        assert _project(tool_name, payload) == {
            "id": "d1",
            "fname": "leave.pdf",
            "source_document_id": None,
            "inserted_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }


class TestPagination:
    def test_exactly_the_list_tools_are_paginated_by_limit(self) -> None:
        paginated = {
            tool.name: tool.page_size_param
            for tool in READ_ONLY_TOOLS
            if tool.page_size_param is not None
        }

        assert paginated == {name: "limit" for name in PAGINATED_TOOLS}

    @pytest.mark.parametrize("tool_name", ["list_collections", "get_collection"])
    def test_collection_tools_are_not_paginated(self, tool_name: str) -> None:
        assert TOOLS_BY_NAME[tool_name].page_size_param is None

    def test_get_collection_document_limit_keeps_its_own_cap(self) -> None:
        args_model = TOOLS_BY_NAME["get_collection"].args_model
        collection_id = "11111111-1111-1111-1111-111111111111"

        assert (
            args_model.model_validate(
                {"collection_id": collection_id, "limit": 100}
            ).limit
            == 100
        )
        with pytest.raises(ValidationError, match="limit"):
            args_model.model_validate({"collection_id": collection_id, "limit": 101})

    @pytest.mark.parametrize("tool_name", PAGINATED_TOOLS)
    def test_page_size_capped_at_max(self, tool_name: str) -> None:
        args_model = TOOLS_BY_NAME[tool_name].args_model

        assert args_model.model_validate({"limit": MAX_PAGE_SIZE}).limit == (
            MAX_PAGE_SIZE
        )
        with pytest.raises(ValidationError, match="limit"):
            args_model.model_validate({"limit": MAX_PAGE_SIZE + 1})
        with pytest.raises(ValidationError, match="limit"):
            args_model.model_validate({"limit": 0})

    @pytest.mark.parametrize(
        "tool_name, offset_param, default_limit",
        [(name, *spec) for name, spec in PAGINATED_TOOLS.items()],
    )
    def test_defaults_and_offset_param_name(
        self, tool_name: str, offset_param: str, default_limit: int
    ) -> None:
        args = TOOLS_BY_NAME[tool_name].args_model.model_validate({})

        assert args.model_dump(include={"limit", offset_param}) == {
            "limit": default_limit,
            offset_param: 0,
        }
        other = "skip" if offset_param == "offset" else "offset"
        assert other not in TOOLS_BY_NAME[tool_name].args_model.model_fields

    @pytest.mark.parametrize("tool_name", PAGINATED_TOOLS)
    def test_route_accepts_probe_limit_at_max_page_size(self, tool_name: str) -> None:
        tool = TOOLS_BY_NAME[tool_name]
        route = _get_routes_by_path()[tool.path]
        [route_limit] = [
            param
            for param in route.dependant.query_params
            if param.alias == tool.page_size_param
        ]

        _, errors = route_limit.validate(
            MAX_PROBE_LIMIT, {}, loc=("query", route_limit.alias)
        )

        assert errors == [], f"{tool_name}: route rejects limit={MAX_PROBE_LIMIT}"

    @pytest.mark.parametrize(
        "tool_name, offset_param",
        [(name, spec[0]) for name, spec in PAGINATED_TOOLS.items()],
    )
    def test_description_explains_paging(
        self, tool_name: str, offset_param: str
    ) -> None:
        description = TOOLS_BY_NAME[tool_name].description

        assert "has_more" in description
        assert f"call again with {offset_param} increased by limit" in description

    def test_prompt_explains_paging_and_lower_bound_counts(self) -> None:
        assert "metadata.has_more" in AGENT_SYSTEM_PROMPT
        assert "skip for documents" in AGENT_SYSTEM_PROMPT
        assert "keep paging until has_more is false" in AGENT_SYSTEM_PROMPT
        assert '"at least N"' in AGENT_SYSTEM_PROMPT
