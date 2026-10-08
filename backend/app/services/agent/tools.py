"""Read-only tool registry for the /agent endpoint.
"""

import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias
from uuid import UUID

from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.services.agent.projection import (
    NUMERIC_MAP,
    SCALAR,
    SCALAR_LIST,
    ProjectionSpec,
    Projector,
    make_projector,
)

HTTP_GET = "GET"

# Enforced by the harness, not the prompt: a tool can never be defined with, or
# send, a method that mutates data.
ALLOWED_TOOL_METHODS = frozenset({HTTP_GET})

_DEFAULT_LIST_LIMIT = 10
_DEFAULT_DATASET_LIST_LIMIT = 20
_MAX_LIST_LIMIT = 100
# The executor sends limit + 1 to probe for a next page; 50 keeps that under the
# routes' le=100 caps.
_MAX_PAGE_SIZE = 50
_PAGE_SIZE_PARAM = "limit"
_OFFSET_PARAM = "offset"
_SKIP_PARAM = "skip"

QueryParamValue: TypeAlias = str | int | float | bool


class _ToolArgs(BaseModel):
    # Unknown keys from the model are rejected rather than silently forwarded.
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    method: str
    path: str
    args_model: type[BaseModel]
    projector: Projector | None = None
    # Sent on every call and not model-settable, e.g. to switch off a route's
    # default-on bulk payload. hash=False: a mapping is unhashable.
    fixed_query_params: Mapping[str, QueryParamValue] = field(
        default_factory=dict, hash=False
    )
    # Args-model field holding the page size; None means the tool isn't paginated.
    page_size_param: str | None = None
    path_param_names: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.method not in ALLOWED_TOOL_METHODS:
            raise ValueError(
                f"[AgentTool] Tool '{self.name}' uses method {self.method}; "
                f"only {sorted(ALLOWED_TOOL_METHODS)} are allowed"
            )

        param_names: set[str] = set()
        for _literal, field_name, _spec, _conversion in string.Formatter().parse(
            self.path
        ):
            if field_name:
                param_names.add(field_name)

        missing = param_names - set(self.args_model.model_fields)
        if missing:
            raise ValueError(
                f"[AgentTool] Tool '{self.name}' path params {sorted(missing)} "
                "are not fields of its args model"
            )
        overlapping = set(self.fixed_query_params) & set(self.args_model.model_fields)
        if overlapping:
            raise ValueError(
                f"[AgentTool] Tool '{self.name}' fixed query params "
                f"{sorted(overlapping)} are also fields of its args model"
            )
        if (
            self.page_size_param is not None
            and self.page_size_param not in self.args_model.model_fields
        ):
            raise ValueError(
                f"[AgentTool] Tool '{self.name}' page_size_param "
                f"'{self.page_size_param}' is not a field of its args model"
            )

        # frozen dataclass: bypass __setattr__ for the derived fields.
        object.__setattr__(self, "path_param_names", frozenset(param_names))
        object.__setattr__(
            self,
            "fixed_query_params",
            MappingProxyType(dict(self.fixed_query_params)),
        )

    def to_anthropic_tool(self) -> ToolParam:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


_SUMMARY_SCORE_SPEC: ProjectionSpec = {
    "name": SCALAR,
    "avg": SCALAR,
    "std": SCALAR,
    "total_pairs": SCALAR,
    "total_items": SCALAR,
    "data_type": SCALAR,
    "distribution": NUMERIC_MAP,
    "unscoreable": NUMERIC_MAP,
}

_STAGE_COST_SPEC: ProjectionSpec = {
    "model": SCALAR,
    "input_tokens": SCALAR,
    "output_tokens": SCALAR,
    "total_tokens": SCALAR,
    "cost_usd": SCALAR,
}

# Run-level aggregates only: per-item traces/scores and the LLM-written
# ai_summary are row-level or free text the model doesn't need.
_EVALUATION_RUN_SPEC: ProjectionSpec = {
    "id": SCALAR,
    "run_name": SCALAR,
    "dataset_id": SCALAR,
    "dataset_name": SCALAR,
    "config_id": SCALAR,
    "config_version": SCALAR,
    "status": SCALAR,
    "run_mode": SCALAR,
    "total_items": SCALAR,
    "is_judge_run": SCALAR,
    "error_message": SCALAR,
    "inserted_at": SCALAR,
    "updated_at": SCALAR,
    "score": {
        "summary_scores": _SUMMARY_SCORE_SPEC,
        "overall": {
            "overall_score": SCALAR,
            "verdict": SCALAR,
            "breakdown": {
                "name": SCALAR,
                "key": SCALAR,
                "score": SCALAR,
                "weight": SCALAR,
                "delta": SCALAR,
                "verdict": SCALAR,
            },
        },
        "category_metrics": {
            "category": SCALAR,
            "total_evals": SCALAR,
            "avg_cosine": SCALAR,
            "avg_correctness": SCALAR,
        },
    },
    "cost": {
        "response": _STAGE_COST_SPEC,
        "embedding": _STAGE_COST_SPEC,
        "judge": _STAGE_COST_SPEC,
        "total_cost_usd": SCALAR,
    },
}

_EVALUATION_DATASET_SPEC: ProjectionSpec = {
    "dataset_id": SCALAR,
    "dataset_name": SCALAR,
    "description": SCALAR,
    "total_items": SCALAR,
    "original_items": SCALAR,
    "duplication_factor": SCALAR,
}

# Only what's needed to chain run → config → knowledge base; the prompt text
# (instructions, prompt_template) stays out.
_CONFIG_VERSION_SPEC: ProjectionSpec = {
    "config_id": SCALAR,
    "version": SCALAR,
    "inserted_at": SCALAR,
    "config_blob": {
        "completion": {
            "type": SCALAR,
            "provider": SCALAR,
            "params": {
                "model": SCALAR,
                "knowledge_base_ids": SCALAR_LIST,
            },
        },
    },
}

_COLLECTION_SPEC: ProjectionSpec = {
    "id": SCALAR,
    "name": SCALAR,
    "description": SCALAR,
    "knowledge_base_id": SCALAR,
    "knowledge_base_provider": SCALAR,
    "inserted_at": SCALAR,
    "updated_at": SCALAR,
}

_COLLECTION_WITH_DOCS_SPEC: ProjectionSpec = {
    **_COLLECTION_SPEC,
    "documents": {
        "id": SCALAR,
        "fname": SCALAR,
        "inserted_at": SCALAR,
    },
}

_DOCUMENT_SPEC: ProjectionSpec = {
    "id": SCALAR,
    "fname": SCALAR,
    "source_document_id": SCALAR,
    "inserted_at": SCALAR,
    "updated_at": SCALAR,
}

_SPEECH_RUN_SPEC: ProjectionSpec = {
    "id": SCALAR,
    "run_name": SCALAR,
    "dataset_id": SCALAR,
    "dataset_name": SCALAR,
    "status": SCALAR,
    "total_items": SCALAR,
    "language_id": SCALAR,
    "models": SCALAR_LIST,
    "error_message": SCALAR,
    "inserted_at": SCALAR,
    "updated_at": SCALAR,
    "score": {
        "summary_scores": _SUMMARY_SCORE_SPEC,
    },
}

_SPEECH_DATASET_SPEC: ProjectionSpec = {
    "id": SCALAR,
    "name": SCALAR,
    "description": SCALAR,
    "language_id": SCALAR,
    "dataset_metadata": {
        "sample_count": SCALAR,
        "has_ground_truth_count": SCALAR,
    },
    "inserted_at": SCALAR,
    "updated_at": SCALAR,
}

_project_evaluation_run = make_projector(_EVALUATION_RUN_SPEC)
_project_evaluation_dataset = make_projector(_EVALUATION_DATASET_SPEC)
_project_config_version = make_projector(_CONFIG_VERSION_SPEC)
_project_collection = make_projector(_COLLECTION_SPEC)
_project_collection_with_docs = make_projector(_COLLECTION_WITH_DOCS_SPEC)
_project_document = make_projector(_DOCUMENT_SPEC)
_project_speech_run = make_projector(_SPEECH_RUN_SPEC)
_project_speech_dataset = make_projector(_SPEECH_DATASET_SPEC)

# The speech get routes default these to true; the agent never wants
# per-sample payloads, so they are pinned off rather than left to the model.
_EXCLUDE_RESULTS: Mapping[str, QueryParamValue] = {"include_results": False}
_EXCLUDE_SAMPLES: Mapping[str, QueryParamValue] = {"include_samples": False}


class _NoArgs(_ToolArgs):
    pass


class _ListEvaluationRunsArgs(_ToolArgs):
    limit: int = Field(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)
    dataset_id: int | None = Field(
        default=None, description="Only runs of this evaluation dataset"
    )


class _EvaluationRunArgs(_ToolArgs):
    evaluation_id: int = Field(description="Evaluation run id")


class _ListDatasetsArgs(_ToolArgs):
    limit: int = Field(default=_DEFAULT_DATASET_LIST_LIMIT, ge=1, le=_MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)


class _DatasetArgs(_ToolArgs):
    dataset_id: int = Field(description="Dataset id")


class _ConfigVersionArgs(_ToolArgs):
    config_id: UUID = Field(description="Config id (UUID), e.g. a run's config_id")
    version_number: int = Field(
        ge=1, description="Version number, e.g. a run's config_version"
    )


class _CollectionArgs(_ToolArgs):
    collection_id: UUID = Field(description="Collection id (UUID)")
    include_docs: bool = Field(
        default=True, description="Include the documents linked to the collection"
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=_MAX_LIST_LIMIT,
        description="Maximum number of documents to include",
    )


class _ListDocumentsArgs(_ToolArgs):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=_DEFAULT_DATASET_LIST_LIMIT, ge=1, le=_MAX_PAGE_SIZE)


class _DocumentArgs(_ToolArgs):
    doc_id: UUID = Field(description="Document id (UUID)")


class _ListSpeechRunsArgs(_ToolArgs):
    limit: int = Field(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)
    dataset_id: int | None = Field(
        default=None, description="Only runs of this dataset"
    )
    status: str | None = Field(
        default=None,
        max_length=50,
        description="Only runs in this status, e.g. 'completed', 'failed'",
    )


class _SttRunArgs(_ToolArgs):
    run_id: int = Field(description="STT evaluation run id")


class _SttDatasetArgs(_ToolArgs):
    dataset_id: int = Field(description="STT dataset id")


class _TtsRunArgs(_ToolArgs):
    run_id: int = Field(description="TTS evaluation run id")


_API = settings.API_V1_STR


def _paging_hint(offset_param: str) -> str:
    return (
        f"Results come newest first in pages of {_PAGE_SIZE_PARAM}; when "
        f"metadata.has_more is true, call again with {offset_param} increased by "
        f"{_PAGE_SIZE_PARAM}."
    )


READ_ONLY_TOOLS: tuple[AgentTool, ...] = (
    AgentTool(
        name="list_evaluation_runs",
        description=(
            "List the project's text evaluation runs, newest first. Each run has id, "
            "run_name, dataset_id/dataset_name, status, run_mode, total_items, "
            "config_id + config_version, summary scores, cost, error_message and "
            "timestamps. Filter by dataset_id if given. " + _paging_hint(_OFFSET_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations",
        args_model=_ListEvaluationRunsArgs,
        projector=_project_evaluation_run,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_evaluation_run",
        description=(
            "Get one text evaluation run by id: status, summary scores, overall "
            "score breakdown, per-category metrics, cost, error_message and the "
            "config_id + config_version it ran with. Use config_id/config_version "
            "with get_config_version to see the model and knowledge bases used."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/{{evaluation_id}}",
        args_model=_EvaluationRunArgs,
        projector=_project_evaluation_run,
    ),
    AgentTool(
        name="list_evaluation_datasets",
        description=(
            "List the project's text evaluation datasets: dataset_id, "
            "dataset_name, description, total_items, original_items, "
            "duplication_factor. " + _paging_hint(_OFFSET_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/datasets",
        args_model=_ListDatasetsArgs,
        projector=_project_evaluation_dataset,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_evaluation_dataset",
        description=(
            "Get one text evaluation dataset by id (item counts, description). "
            "Use list_evaluation_runs with dataset_id to find runs over it."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/datasets/{{dataset_id}}",
        args_model=_DatasetArgs,
        projector=_project_evaluation_dataset,
    ),
    AgentTool(
        name="get_config_version",
        description=(
            "Get one version of a config: its completion type, provider, model "
            "and knowledge_base_ids (under config_blob.completion.params). Match "
            "knowledge_base_ids against the knowledge_base_id of collections from "
            "list_collections to find which knowledge base (and documents) a "
            "config used."
        ),
        method=HTTP_GET,
        path=f"{_API}/configs/{{config_id}}/versions/{{version_number}}",
        args_model=_ConfigVersionArgs,
        projector=_project_config_version,
    ),
    AgentTool(
        name="list_collections",
        description=(
            "List all of the project's knowledge-base collections: id, name, "
            "description, knowledge_base_id (the vector store id referenced by "
            "configs' knowledge_base_ids) and knowledge_base_provider."
        ),
        method=HTTP_GET,
        path=f"{_API}/collections",
        args_model=_NoArgs,
        projector=_project_collection,
    ),
    AgentTool(
        name="get_collection",
        description=(
            "Get one collection by id, including the documents (id, fname, "
            "inserted_at) in it when include_docs is true. Get the collection id "
            "from list_collections."
        ),
        method=HTTP_GET,
        path=f"{_API}/collections/{{collection_id}}",
        args_model=_CollectionArgs,
        projector=_project_collection_with_docs,
    ),
    AgentTool(
        name="list_documents",
        description=(
            "List the project's uploaded documents: id, fname, "
            "source_document_id for transformed documents, timestamps. "
            + _paging_hint(_SKIP_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/documents",
        args_model=_ListDocumentsArgs,
        projector=_project_document,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_document",
        description="Get one document by id: fname, source_document_id, timestamps.",
        method=HTTP_GET,
        path=f"{_API}/documents/{{doc_id}}",
        args_model=_DocumentArgs,
        projector=_project_document,
    ),
    AgentTool(
        name="list_stt_evaluation_runs",
        description=(
            "List speech-to-text (STT) evaluation runs: id, run_name, "
            "dataset, models, language_id, status, total_items, summary scores. "
            + _paging_hint(_OFFSET_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/runs",
        args_model=_ListSpeechRunsArgs,
        projector=_project_speech_run,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_stt_evaluation_run",
        description=(
            "Get one STT evaluation run by id: status, models, total_items, "
            "error_message and run-level summary scores."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/runs/{{run_id}}",
        args_model=_SttRunArgs,
        projector=_project_speech_run,
        fixed_query_params=_EXCLUDE_RESULTS,
    ),
    AgentTool(
        name="list_stt_evaluation_datasets",
        description=(
            "List STT evaluation datasets: id, name, description, language_id, "
            "dataset_metadata (sample_count, has_ground_truth_count). "
            + _paging_hint(_OFFSET_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/datasets",
        args_model=_ListDatasetsArgs,
        projector=_project_speech_dataset,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_stt_evaluation_dataset",
        description=(
            "Get one STT dataset by id: name, description, language_id and "
            "dataset_metadata (sample_count, has_ground_truth_count)."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/datasets/{{dataset_id}}",
        args_model=_SttDatasetArgs,
        projector=_project_speech_dataset,
        fixed_query_params=_EXCLUDE_SAMPLES,
    ),
    AgentTool(
        name="list_tts_evaluation_runs",
        description=(
            "List text-to-speech (TTS) evaluation runs: id, run_name, "
            "dataset, models, language_id, status, total_items, summary scores. "
            + _paging_hint(_OFFSET_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/runs",
        args_model=_ListSpeechRunsArgs,
        projector=_project_speech_run,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_tts_evaluation_run",
        description=(
            "Get one TTS evaluation run by id: status, models, total_items, "
            "error_message and run-level summary scores."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/runs/{{run_id}}",
        args_model=_TtsRunArgs,
        projector=_project_speech_run,
        fixed_query_params=_EXCLUDE_RESULTS,
    ),
    AgentTool(
        name="list_tts_evaluation_datasets",
        description=(
            "List TTS evaluation datasets: id, name, description, language_id, "
            "dataset_metadata (sample_count). " + _paging_hint(_OFFSET_PARAM)
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/datasets",
        args_model=_ListDatasetsArgs,
        projector=_project_speech_dataset,
        page_size_param=_PAGE_SIZE_PARAM,
    ),
    AgentTool(
        name="get_tts_evaluation_dataset",
        description=(
            "Get one TTS dataset by id: name, description, language_id and "
            "dataset_metadata (sample_count)."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/datasets/{{dataset_id}}",
        args_model=_DatasetArgs,
        projector=_project_speech_dataset,
    ),
)


def _require_projectors(tools: tuple[AgentTool, ...]) -> None:
    # Allowlisting is the only filter between route output and the model, so an
    # unprojected registry tool would forward the full response.
    for tool in tools:
        if tool.projector is None:
            raise ValueError(
                f"[_require_projectors] Tool '{tool.name}' has no projector"
            )


_require_projectors(READ_ONLY_TOOLS)

TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in READ_ONLY_TOOLS}

# Sent on every turn (even when tools are disabled) so the cached prefix is stable.
ANTHROPIC_TOOL_DEFINITIONS: list[ToolParam] = [
    tool.to_anthropic_tool() for tool in READ_ONLY_TOOLS
]
