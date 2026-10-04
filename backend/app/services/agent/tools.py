"""Read-only tool registry for the /agent endpoint.
"""

import string
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.core.config import settings

HTTP_GET = "GET"

# Enforced by the harness, not the prompt: a tool can never be defined with, or
# send, a method that mutates data.
ALLOWED_TOOL_METHODS = frozenset({HTTP_GET})

# Storage locations are internal; signed URLs are never requested, but drop the
# keys anyway so a route change can't start leaking them to the model.
_STORAGE_URL_KEYS = frozenset({"object_store_url", "score_trace_url", "signed_url"})
# Per-item traces inside an eval run's `score`; summary_scores/overall stay.
_SCORE_TRACES_KEY = "traces"

_DEFAULT_LIST_LIMIT = 10
_MAX_LIST_LIMIT = 100

Projector = Callable[[JsonValue], JsonValue]


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
        # frozen dataclass: bypass __setattr__ for the derived field.
        object.__setattr__(self, "path_param_names", frozenset(param_names))

    def to_anthropic_tool(self) -> ToolParam:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


def _drop_keys(value: JsonValue, keys: frozenset[str]) -> JsonValue:
    if isinstance(value, dict):
        cleaned: dict[str, JsonValue] = {}
        for key, item in value.items():
            if key in keys:
                continue
            cleaned[key] = _drop_keys(item, keys)
        return cleaned
    if isinstance(value, list):
        return [_drop_keys(item, keys) for item in value]
    return value


def _strip_storage_urls(data: JsonValue) -> JsonValue:
    return _drop_keys(data, _STORAGE_URL_KEYS)


def _project_evaluation_run(run: JsonValue) -> JsonValue:
    if not isinstance(run, dict):
        return run

    score = run.get("score")
    if isinstance(score, dict):
        summary_score: dict[str, JsonValue] = {}
        for key, value in score.items():
            if key != _SCORE_TRACES_KEY:
                summary_score[key] = value
        run = {**run, "score": summary_score}

    return _strip_storage_urls(run)


def _project_evaluation_runs(runs: JsonValue) -> JsonValue:
    if not isinstance(runs, list):
        return _project_evaluation_run(runs)
    return [_project_evaluation_run(run) for run in runs]


class _NoArgs(_ToolArgs):
    pass


class _ListEvaluationRunsArgs(_ToolArgs):
    limit: int = Field(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_LIST_LIMIT)
    offset: int = Field(default=0, ge=0)
    dataset_id: int | None = Field(
        default=None, description="Only runs of this evaluation dataset"
    )


class _EvaluationRunArgs(_ToolArgs):
    evaluation_id: int = Field(description="Evaluation run id")


class _ListDatasetsArgs(_ToolArgs):
    limit: int = Field(default=20, ge=1, le=_MAX_LIST_LIMIT)
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
    limit: int = Field(default=20, ge=1, le=_MAX_LIST_LIMIT)


class _DocumentArgs(_ToolArgs):
    doc_id: UUID = Field(description="Document id (UUID)")


class _ListSpeechRunsArgs(_ToolArgs):
    limit: int = Field(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_LIST_LIMIT)
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
    include_results: bool = Field(
        default=False,
        description="Include per-sample transcription results (can be large)",
    )
    result_limit: int = Field(default=20, ge=1, le=_MAX_LIST_LIMIT)
    result_offset: int = Field(default=0, ge=0)


class _SttDatasetArgs(_ToolArgs):
    dataset_id: int = Field(description="STT dataset id")
    include_samples: bool = Field(
        default=False, description="Include the dataset's audio samples"
    )
    sample_limit: int = Field(default=20, ge=1, le=_MAX_LIST_LIMIT)
    sample_offset: int = Field(default=0, ge=0)


class _TtsRunArgs(_ToolArgs):
    run_id: int = Field(description="TTS evaluation run id")
    include_results: bool = Field(
        default=False,
        description="Include every per-sample synthesis result (unpaginated; can be large)",
    )


_API = settings.API_V1_STR

READ_ONLY_TOOLS: tuple[AgentTool, ...] = (
    AgentTool(
        name="list_evaluation_runs",
        description=(
            "List the project's text evaluation runs, newest first. Each run has id, "
            "run_name, dataset_id/dataset_name, status, run_mode, total_items, "
            "config_id + config_version, summary scores, cost, error_message and "
            "timestamps. Paginate with limit/offset; to count runs, keep paging "
            "until a page is shorter than limit. Filter by dataset_id if given."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations",
        args_model=_ListEvaluationRunsArgs,
        projector=_project_evaluation_runs,
    ),
    AgentTool(
        name="get_evaluation_run",
        description=(
            "Get one text evaluation run by id: status, summary scores, cost, "
            "error_message and the config_id + config_version it ran with. Use "
            "config_id/config_version with get_config_version to see the prompt, "
            "model and knowledge bases used."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/{{evaluation_id}}",
        args_model=_EvaluationRunArgs,
        projector=_project_evaluation_run,
    ),
    AgentTool(
        name="list_evaluation_datasets",
        description=(
            "List the project's text evaluation datasets, newest first: dataset_id, "
            "dataset_name, description, total_items, original_items, "
            "duplication_factor. Paginate with limit/offset."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/datasets",
        args_model=_ListDatasetsArgs,
        projector=_strip_storage_urls,
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
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="get_config_version",
        description=(
            "Get one version of a config: its config_blob (provider, model, "
            "instructions/prompt and params) and commit_message. The blob's "
            "completion.params may carry knowledge_base_ids; match those against "
            "the knowledge_base_id of collections from list_collections to find "
            "which knowledge base (and documents) a config used."
        ),
        method=HTTP_GET,
        path=f"{_API}/configs/{{config_id}}/versions/{{version_number}}",
        args_model=_ConfigVersionArgs,
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
    ),
    AgentTool(
        name="get_collection",
        description=(
            "Get one collection by id, including the documents (id, fname, "
            "timestamps) in it when include_docs is true. Get the collection id "
            "from list_collections."
        ),
        method=HTTP_GET,
        path=f"{_API}/collections/{{collection_id}}",
        args_model=_CollectionArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="list_documents",
        description=(
            "List the project's uploaded documents, newest first: id, fname, "
            "source_document_id for transformed documents, timestamps. Paginate "
            "with skip/limit; metadata.has_more says whether more pages exist."
        ),
        method=HTTP_GET,
        path=f"{_API}/documents",
        args_model=_ListDocumentsArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="get_document",
        description="Get one document by id: fname, source_document_id, timestamps.",
        method=HTTP_GET,
        path=f"{_API}/documents/{{doc_id}}",
        args_model=_DocumentArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="list_stt_evaluation_runs",
        description=(
            "List speech-to-text (STT) evaluation runs, newest first: id, run_name, "
            "dataset, models, status, total_items, score. Paginate with "
            "limit/offset; metadata.total is the exact total matching the filters."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/runs",
        args_model=_ListSpeechRunsArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="get_stt_evaluation_run",
        description=(
            "Get one STT evaluation run by id. Set include_results to also get "
            "per-sample transcriptions, correctness and scores (paginated with "
            "result_limit/result_offset)."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/runs/{{run_id}}",
        args_model=_SttRunArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="list_stt_evaluation_datasets",
        description=(
            "List STT evaluation datasets: id, name, description, language_id, "
            "dataset_metadata. Paginate with limit/offset."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/datasets",
        args_model=_ListDatasetsArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="get_stt_evaluation_dataset",
        description=(
            "Get one STT dataset by id. Set include_samples to also list its "
            "samples with ground-truth transcriptions (paginated)."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/stt/datasets/{{dataset_id}}",
        args_model=_SttDatasetArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="list_tts_evaluation_runs",
        description=(
            "List text-to-speech (TTS) evaluation runs, newest first: id, run_name, "
            "dataset, models, status, total_items, score. Paginate with "
            "limit/offset; metadata.total is the exact total matching the filters."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/runs",
        args_model=_ListSpeechRunsArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="get_tts_evaluation_run",
        description=(
            "Get one TTS evaluation run by id. include_results adds every "
            "per-sample result (not paginated, so only use it when needed)."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/runs/{{run_id}}",
        args_model=_TtsRunArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="list_tts_evaluation_datasets",
        description=(
            "List TTS evaluation datasets: id, name, description, language_id, "
            "dataset_metadata. Paginate with limit/offset."
        ),
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/datasets",
        args_model=_ListDatasetsArgs,
        projector=_strip_storage_urls,
    ),
    AgentTool(
        name="get_tts_evaluation_dataset",
        description="Get one TTS dataset by id: name, description, dataset_metadata.",
        method=HTTP_GET,
        path=f"{_API}/evaluations/tts/datasets/{{dataset_id}}",
        args_model=_DatasetArgs,
        projector=_strip_storage_urls,
    ),
)

TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in READ_ONLY_TOOLS}

# Sent on every turn (even when tools are disabled) so the cached prefix is stable.
ANTHROPIC_TOOL_DEFINITIONS: list[ToolParam] = [
    tool.to_anthropic_tool() for tool in READ_ONLY_TOOLS
]
