"""BATCH API-client staged pipeline: stage build/submit, verdict parsing, advance.

One assessment -> an ordered series of provider batches: all GATE pre-filters first
(config order), then all PASS-THROUGH pre-filters, then the assessment stage. Runtime
state lives entirely in ``assessment.execution`` (``AssessmentExecution``).

Conditional forwarding:
  - a GATE stage runs on every row; rows whose verdict fails are marked
    ``gate_passed=False`` but still flow through the remaining pass-through stages
    so their metadata is complete.
  - a PASS-THROUGH stage runs on every row and never changes ``gate_passed``.
  - the assessment stage batches ONLY gate-passed rows; gate-failed rows get
    ``response=null`` in the result, carrying their pre-filter verdicts.
"""

import json
import logging
from typing import Any, cast
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session, col, select

from app.core.batch import (
    BATCH_KEY,
    AnthropicBatchProvider,
    BatchJobState,
    GeminiBatchProvider,
    GoogleGCPBatchProvider,
    MessageBatchStatus,
    OpenAIBatchProvider,
    extract_text_from_response_dict,
    poll_batch_status,
    process_completed_batch,
    start_batch_job,
)
from app.core.batch.base import BatchProvider
from app.core.batch.client import GeminiClient
from app.core.config import settings
from app.core.db import engine
from app.crud.assessment import api
from app.crud.assessment.batch import (
    build_anthropic_jsonl,
    build_google_jsonl,
    build_openai_jsonl,
)
from app.crud.credentials import get_provider_credential
from app.crud.job import get_batch_job
from app.models.assessment import (
    ApiStage,
    Assessment,
    AssessmentAttachment,
    AssessmentExecution,
    AssessmentStatus,
    BatchInput,
    ParsedResult,
    PipelineStep,
    PreFilterVerdict,
    StageCounters,
    StageKind,
    StageStatus,
)
from app.models.batch_job import BatchJob, BatchJobType
from app.models.config.assessment_blob import (
    ATTACHMENT_COLUMN_TYPES,
    AssessmentConfigBlob,
    AssessmentPreFilters,
    TopicRelevanceFilter,
)
from app.models.llm.constants import DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS
from app.services.llm.mappers import (
    map_kaapi_to_anthropic_params,
    map_kaapi_to_google_params,
    map_kaapi_to_openai_params,
)
from app.services.assessment.utils.attachments import rewrite_gcs_attachment_urls
from app.services.assessment.validators import normalize_llm_text, stage_batch_prefix
from app.services.llm.providers.registry import LLMProvider
from app.utils import (
    get_anthropic_client,
    get_openai_client,
)

logger = logging.getLogger(__name__)


def _google_gcp_credential(
    *, session: Session, organization_id: int, project_id: int
) -> dict[str, Any]:
    """Vertex needs the SA key + bucket; a missing credential is client-fixable."""
    cred = get_provider_credential(
        session=session,
        provider=LLMProvider.GOOGLE_GCP,
        project_id=project_id,
        org_id=organization_id,
    )
    if not isinstance(cred, dict):
        raise HTTPException(
            status_code=404,
            detail="google-gcp credentials not configured for this project",
        )
    return cred


# Re-poll cadence for a stage's provider batch, mirroring the assessment cron interval.
POLL_COUNTDOWN_SECONDS = settings.CRON_INTERVAL_MINUTES * 60


# Structured-output schema injected into every pre-filter stage's completion; the
# verdict parser reads ``verdict``. Both keys are required because provider strict
# JSON mode (OpenAI) rejects optional properties — ``reasoning`` may be empty.
PREFILTER_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}

_PREFILTER_INSTRUCTION = (
    "You are a pre-filter gate. Judge the item against the criteria in the prompt. "
    "Return verdict=true if it satisfies the criteria, else verdict=false, with brief "
    "reasoning."
)

_SUCCESS_STATUSES = {
    "completed",
    BatchJobState.SUCCEEDED.value,
    MessageBatchStatus.ENDED.value,
}
_FAILED_STATUSES = {
    "failed",
    "expired",
    "cancelled",
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}

_SUPPORTED_PROVIDERS = {
    LLMProvider.OPENAI,
    LLMProvider.GOOGLE,
    LLMProvider.GOOGLE_AISTUDIO,
    LLMProvider.GOOGLE_GCP,
    LLMProvider.ANTHROPIC,
}


def is_supported_provider(provider_name: str) -> bool:
    return provider_name in _SUPPORTED_PROVIDERS


def build_pipeline(pre_filters: AssessmentPreFilters | None) -> list[PipelineStep]:
    """Ordered stages: GATE pre-filters, then PASS-THROUGH pre-filters, then assessment."""
    present: list[tuple[ApiStage, TopicRelevanceFilter]] = []
    if pre_filters is not None:
        if pre_filters.topic_relevance is not None:
            present.append((ApiStage.TOPIC_RELEVANCE, pre_filters.topic_relevance))

    gates = [
        PipelineStep(stage=stage, kind=StageKind.GATE)
        for stage, flt in present
        if flt.stop_on_fail
    ]
    passthrough = [
        PipelineStep(stage=stage, kind=StageKind.PASS_THROUGH)
        for stage, flt in present
        if not flt.stop_on_fail
    ]
    return [
        *gates,
        *passthrough,
        PipelineStep(stage=ApiStage.ASSESSMENT, kind=StageKind.ASSESSMENT),
    ]


def column_kinds(
    columns: list[str], input_columns: dict[str, Any]
) -> tuple[list[str], list[AssessmentAttachment]]:
    """Split columns into text names and attachment specs per the config's ``input_schema``.

    A column typed image/pdf/video is an attachment (url-format only); anything else is text.
    """
    text_columns: list[str] = []
    attachments: list[AssessmentAttachment] = []
    for column in columns:
        spec = input_columns.get(column) or {}
        col_type = spec.get("type", "text")
        if col_type in ATTACHMENT_COLUMN_TYPES:
            if (spec.get("format") or "url") != "url":
                raise ValueError(
                    f"BATCH attachment column '{column}' must be url-format; base64 is "
                    f"not supported for batch submission."
                )
            attachments.append(
                AssessmentAttachment(column=column, type=col_type, format="url")
            )
        else:
            text_columns.append(column)
    return text_columns, attachments


def build_rows(
    batch_input: BatchInput,
    input_columns: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[str], list[AssessmentAttachment]]:
    """Normalise submissions to string cells over the union of row and schema columns.

    Submit-time only; the batch task streams the stored rows instead. A column a row omits
    (non-strict) is filled with "".
    """
    submissions = batch_input.data or []
    input_columns = input_columns or {}
    columns = list(
        dict.fromkeys([*(key for sub in submissions for key in sub), *input_columns])
    )
    text_columns, attachments = column_kinds(columns, input_columns)
    rows = [{col: str(sub.get(col, "")) for col in columns} for sub in submissions]
    return rows, text_columns, attachments


def _stage_prompt(blob: AssessmentConfigBlob, stage: ApiStage) -> str | None:
    """Per-row prompt template for a stage, sourced from the config's ``submission``.

    The assessment carries a mandatory ``submission``; a pre-filter may carry its own
    optional one (None when it declares none — its criteria then live entirely in
    params.instructions with the item's columns + attachments as the user content).
    """
    if stage == ApiStage.ASSESSMENT:
        return blob.assessment.params["submission"]
    return cast(
        "str | None", _prefilter_for_stage(blob, stage).params.get("submission")
    )


def _prefilter_for_stage(
    blob: AssessmentConfigBlob, stage: ApiStage
) -> TopicRelevanceFilter:
    """The pre-filter config object for a pre-filter stage (topic_relevance)."""
    pre = blob.pre_filters
    if pre is not None:
        if stage == ApiStage.TOPIC_RELEVANCE and pre.topic_relevance is not None:
            return pre.topic_relevance
    raise ValueError(
        f"[_prefilter_for_stage] No pre-filter configured for stage {stage}"
    )


def _stage_params(blob: AssessmentConfigBlob, stage: ApiStage) -> dict[str, Any]:
    """Kaapi params for a stage: the assessment uses its own params + output schema; each
    pre-filter uses ITS own params (criteria in params.instructions), with the verdict
    schema + gate directive layered on.
    """
    if stage == ApiStage.ASSESSMENT:
        params = dict(blob.assessment.params)
        json_schema = params.pop("json_output_schema", None)
        # submission is a prompt concern, not a provider param.
        params.pop("submission", None)
        if json_schema is not None:
            params["output_schema"] = json_schema  # provider param name
        # Normalised here, not in the mapper, so only this path's prompts are rewritten.
        params["instructions"] = normalize_llm_text(params.get("instructions") or "")
        return params

    flt = _prefilter_for_stage(blob, stage)
    params = dict(flt.params)
    params.pop("submission", None)  # prompt template, not a provider param
    params["output_schema"] = PREFILTER_VERDICT_SCHEMA
    # The config's criteria (mandatory params.instructions) is the system prompt;
    # append the gate directive so the model returns the verdict+reasoning contract.
    criteria = normalize_llm_text(params.get("instructions") or "")
    params["instructions"] = f"{criteria}\n\n{_PREFILTER_INSTRUCTION}"
    return params


def stage_provider_model(
    blob: AssessmentConfigBlob, stage: ApiStage
) -> tuple[str, str]:
    """Provider + model for a stage: each pre-filter uses its own; assessment uses the config's."""
    if stage == ApiStage.TOPIC_RELEVANCE:
        flt = _prefilter_for_stage(blob, stage)
        return flt.provider, flt.params["model"]
    return blob.assessment.provider, blob.assessment.params["model"]


def resolve_blob(session: Session, assessment: Assessment) -> AssessmentConfigBlob:
    """The pinned config version's blob; raises on a deleted version or an old shape."""
    from app.crud.config.version import ConfigVersionCrud
    from app.models.config.config import ConfigTag

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=assessment.config_id,
        project_id=assessment.project_id,
        tag=ConfigTag.ASSESSMENT,
    )
    version = version_crud.exists_or_raise(version_number=assessment.config_version)
    return AssessmentConfigBlob.model_validate(version.config_blob)


def _submit_provider_batch(
    *,
    session: Session,
    provider_name: str,
    model: str,
    rows: list[dict[str, str]],
    text_columns: list[str],
    attachments: list[AssessmentAttachment],
    prompt: str | None,
    params: dict[str, Any],
    row_indices: list[int],
    organization_id: int,
    project_id: int,
    description: str,
) -> BatchJob:
    """Build provider JSONL and submit it via the shared batch infra."""
    # Resolve gs:// attachments to provider-reachable URLs before building JSONL.
    rows = rewrite_gcs_attachment_urls(
        session=session,
        rows=rows,
        attachments=attachments,
        llm_provider=provider_name,
        project_id=project_id,
        organization_id=organization_id,
    )

    if provider_name == LLMProvider.OPENAI:
        mapped, _ = map_kaapi_to_openai_params(session=session, kaapi_params=params)
        jsonl = build_openai_jsonl(
            rows, text_columns, attachments, prompt, mapped, row_indices
        )
        provider: BatchProvider = OpenAIBatchProvider(
            client=get_openai_client(
                session=session, org_id=organization_id, project_id=project_id
            )
        )
        config = {
            "endpoint": "/v1/responses",
            "description": description,
            "completion_window": "24h",
        }
    elif provider_name in (
        LLMProvider.GOOGLE,
        LLMProvider.GOOGLE_AISTUDIO,
        LLMProvider.GOOGLE_GCP,
    ):
        mapped, _ = map_kaapi_to_google_params(params)
        jsonl = build_google_jsonl(
            rows, text_columns, attachments, prompt, mapped, row_indices
        )
        if provider_name == LLMProvider.GOOGLE_GCP:
            cred = _google_gcp_credential(
                session=session,
                organization_id=organization_id,
                project_id=project_id,
            )
            provider = GoogleGCPBatchProvider.from_credentials(cred, model=model)
            config = {"display_name": description}  # Vertex uses a bare model id
        else:
            gemini = GeminiClient.from_credentials(
                session=session, org_id=organization_id, project_id=project_id
            )
            provider = GeminiBatchProvider(
                client=gemini.client, model=f"models/{model}"
            )
            config = {"display_name": description, "model": f"models/{model}"}

    elif provider_name == LLMProvider.ANTHROPIC:
        mapped, _ = map_kaapi_to_anthropic_params(params)
        jsonl = build_anthropic_jsonl(
            rows, text_columns, attachments, prompt, mapped, row_indices
        )
        provider = AnthropicBatchProvider(
            client=get_anthropic_client(
                session=session, org_id=organization_id, project_id=project_id
            )
        )
        config = {
            "model": mapped.get("model"),
            "max_tokens": mapped.get("max_tokens")
            or DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS,
        }
    else:
        raise ValueError(
            f"[_submit_provider_batch] Unsupported provider {provider_name}"
        )

    if not jsonl:
        raise ValueError(
            f"[_submit_provider_batch] No batch rows built for stage description={description}"
        )

    return start_batch_job(
        session=session,
        provider=provider,
        provider_name=provider_name,
        job_type=BatchJobType.ASSESSMENT,
        organization_id=organization_id,
        project_id=project_id,
        jsonl_data=jsonl,
        config=config,
    )


def _build_batch_provider(
    *, session: Session, provider_name: str, organization_id: int, project_id: int
) -> BatchProvider:
    if provider_name == LLMProvider.OPENAI:
        return OpenAIBatchProvider(
            client=get_openai_client(
                session=session, org_id=organization_id, project_id=project_id
            )
        )
    if provider_name in (
        LLMProvider.GOOGLE,
        LLMProvider.GOOGLE_AISTUDIO,
        LLMProvider.GOOGLE_GCP,
    ):
        if provider_name == LLMProvider.GOOGLE_GCP:
            cred = _google_gcp_credential(
                session=session,
                organization_id=organization_id,
                project_id=project_id,
            )
            return GoogleGCPBatchProvider.from_credentials(cred)
        gemini = GeminiClient.from_credentials(
            session=session, org_id=organization_id, project_id=project_id
        )
        return GeminiBatchProvider(client=gemini.client)
    if provider_name == LLMProvider.ANTHROPIC:
        return AnthropicBatchProvider(
            client=get_anthropic_client(
                session=session, org_id=organization_id, project_id=project_id
            )
        )
    raise ValueError(f"[_build_batch_provider] Unsupported provider {provider_name}")


def _row_index(row_id: Any) -> int | None:
    key = str(row_id or "")
    prefix, _, suffix = key.partition("_")
    if prefix != "row":
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def _err_str(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


def _openai_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    chunks: list[str] = []
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if (
                        isinstance(content, dict)
                        and content.get("type") == "output_text"
                        and isinstance(content.get("text"), str)
                    ):
                        chunks.append(content["text"])
    return "".join(chunks)


def _parse_one(result: dict[str, Any], provider_name: str) -> ParsedResult:
    error = result.get("error")
    if error:
        return {
            "output": None,
            "error": _err_str(error),
            "usage": None,
            "response_id": None,
        }

    if provider_name == LLMProvider.OPENAI:
        response = result.get("response") or {}
        status = response.get("status_code")
        body = response.get("body") or {}
        if status and status >= 400:
            return {
                "output": None,
                "error": (body.get("error") or {}).get("message", f"status {status}"),
                "usage": None,
                "response_id": body.get("id"),
            }
        text = body.get("output_text") or _openai_output_text(body.get("output"))
        return {
            "output": text or None,
            "error": None if text else "Empty response output",
            "usage": body.get("usage"),
            "response_id": body.get("id"),
        }

    if provider_name == LLMProvider.ANTHROPIC:
        response = result.get("response") or {}
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        )
        return {
            "output": text or None,
            "error": None if text else "Empty response",
            "usage": response.get("usage"),
            "response_id": response.get("id"),
        }

    if provider_name in (
        LLMProvider.GOOGLE,
        LLMProvider.GOOGLE_AISTUDIO,
        LLMProvider.GOOGLE_GCP,
    ):
        response = result.get("response")
        text = extract_text_from_response_dict(response) if response else None
        return {
            "output": text or None,
            "error": None if text else "Empty response",
            "usage": None,
            "response_id": None,
        }

    return {
        "output": None,
        "error": f"Unknown provider {provider_name}",
        "usage": None,
        "response_id": None,
    }


def parse_batch_results(
    raw_results: list[dict[str, Any]], provider_name: str
) -> dict[int, ParsedResult]:
    """Raw provider results -> {row_index: {output, error, usage, response_id}}."""
    parsed: dict[int, ParsedResult] = {}
    for result in raw_results:
        idx = _row_index(result.get(BATCH_KEY) or result.get("key"))
        if idx is None:
            continue
        parsed[idx] = _parse_one(result, provider_name)
    return parsed


def _parse_verdict(output: str | None) -> PreFilterVerdict:
    """Read a pre-filter verdict. Unparseable output fails open (verdict=True)."""
    if not output:
        return PreFilterVerdict(verdict=True)
    try:
        data = json.loads(output)
        return PreFilterVerdict(
            verdict=bool(data.get("verdict", True)),
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning(
            "[_parse_verdict] Unparseable verdict, failing open | output=%s",
            output[:200],
        )
        return PreFilterVerdict(verdict=True)


def _record_stage(
    bag: AssessmentExecution,
    stage: ApiStage,
    kind: StageKind,
    parsed: dict[int, ParsedResult],
) -> None:
    """Fold a completed stage's parsed results into the bag per its kind."""
    if kind == StageKind.ASSESSMENT:
        return  # assessment outputs are read from object store at result time

    verdicts: dict[int, PreFilterVerdict] = {}
    passed_count = 0
    for idx, out in parsed.items():
        verdict = _parse_verdict(out.get("output"))
        verdicts[idx] = verdict
        if verdict.verdict:
            passed_count += 1
        elif kind == StageKind.GATE:
            bag.gate_passed[idx] = False

    bag.verdicts[stage] = verdicts
    bag.counters[stage] = StageCounters(
        total=len(parsed), passed=passed_count, rejected=len(parsed) - passed_count
    )


def _row_subset(bag: AssessmentExecution, kind: StageKind, total: int) -> list[int]:
    if kind == StageKind.ASSESSMENT:
        return [i for i in range(total) if bag.gate_passed[i]]
    return list(range(total))


def _submit_stage(
    *,
    session: Session,
    assessment: Assessment,
    blob: AssessmentConfigBlob,
    bag: AssessmentExecution,
    stage: ApiStage,
    organization_id: int,
    project_id: int,
) -> bool:
    """Build + submit the current stage's batch on its row subset. Returns success.

    A stage already in flight is reported as submitted (no second provider batch): the
    in-memory bag can predate a concurrent task's write, so the row is re-read here.
    Rows are streamed from storage once per stage submit; only this stage's subset is held.
    """
    session.refresh(assessment)
    persisted = api.load_execution_state(assessment)
    in_flight_batch_id = persisted.stage_batches.get(stage) if persisted else None
    if (
        persisted is not None
        and in_flight_batch_id is not None
        and persisted.stage_status == StageStatus.PROCESSING
    ):
        # Deliberately leaves `bag` unpersisted: the stored state is the fresher one.
        logger.warning(
            "[_submit_stage] Stage already submitted, skipping duplicate | "
            "assessment_id=%s | stage=%s | batch_job=%s",
            assessment.id,
            stage,
            in_flight_batch_id,
        )
        return True

    from app.services.assessment.api.submission_store import open_submission_rows

    kind = bag.stage_kind(stage)
    input_columns = {
        name: col.model_dump(exclude_none=True)
        for name, col in blob.input_schema.items()
    }
    text_columns, attachments = column_kinds(list(input_columns), input_columns)
    subset = _row_subset(bag, kind, assessment.total_items)

    if not subset:
        # No rows left for this stage (everything gated out upstream). Persist the
        # empty counters and return False so the caller advances/finalizes instead of
        # re-submitting a PENDING stage forever.
        logger.info(
            "[_submit_stage] Empty subset, skipping | assessment_id=%s | stage=%s",
            assessment.id,
            stage,
        )
        bag.counters[stage] = StageCounters()
        api.save_execution_state(session=session, assessment=assessment, state=bag)
        return False

    wanted = set(subset)
    with open_submission_rows(session=session, assessment=assessment) as stream:
        rows = [row for idx, row in enumerate(stream) if idx in wanted]

    provider_name, model = stage_provider_model(blob, stage)
    batch_job = _submit_provider_batch(
        session=session,
        provider_name=provider_name,
        model=model,
        rows=rows,
        text_columns=text_columns,
        attachments=attachments,
        prompt=_stage_prompt(blob, stage),
        params=_stage_params(blob, stage),
        row_indices=subset,
        organization_id=organization_id,
        project_id=project_id,
        description=f"assessment-{assessment.id}-{stage}",
    )
    if batch_job.id is None:
        raise ValueError("[_submit_stage] Batch job was not persisted")

    bag.stage = stage
    bag.stage_status = StageStatus.PROCESSING
    bag.stage_batches[stage] = batch_job.id
    api.save_execution_state(session=session, assessment=assessment, state=bag)
    logger.info(
        "[_submit_stage] Submitted | assessment_id=%s | stage=%s | batch_job=%s | rows=%s",
        assessment.id,
        stage,
        batch_job.id,
        len(subset),
    )
    return True


def error_file_entries(provider: BatchProvider, file_id: str) -> list[dict[str, Any]]:
    """Parsed lines of a provider error file (OpenAI only); an unparseable line is skipped."""
    entries: list[dict[str, Any]] = []
    for line in provider.download_file(file_id).splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(
                "[error_file_entries] Unparseable line skipped | file_id=%s", file_id
            )
    return entries


def _poll_outcome(
    session: Session,
    provider: BatchProvider,
    batch_job: BatchJob,
    assessment_id: UUID,
) -> tuple[str, list[dict[str, Any]] | None]:
    """Poll a stage batch. Returns ('processing'|'completed'|'failed', results)."""
    status_result = poll_batch_status(
        session=session, provider=provider, batch_job=batch_job
    )
    session.refresh(batch_job)
    status = batch_job.provider_status

    if status in _SUCCESS_STATUSES:
        counts = status_result.get("request_counts") or {}
        if counts.get("completed", 0) == 0 and (
            counts.get("failed", 0) > 0
            or status_result.get("error_file_id")
            or status_result.get("error_message")
        ):
            return "failed", None
        if batch_job.provider_output_file_id:
            results, _ = process_completed_batch(
                session=session,
                provider=provider,
                batch_job=batch_job,
                subdirectory=stage_batch_prefix(assessment_id, batch_job.id),
            )
            # Rows OpenAI rejected live only in the error file; without them they would
            # read as "no output, no error" and the stage could finish COMPLETED.
            if batch_job.provider_error_file_id:
                try:
                    results = [
                        *results,
                        *error_file_entries(provider, batch_job.provider_error_file_id),
                    ]
                except Exception:
                    logger.warning(
                        "[_poll_outcome] Provider error file unreadable | batch_job_id=%s",
                        batch_job.id,
                        exc_info=True,
                    )
            return "completed", results
        return "processing", None  # output not ready yet
    if status in _FAILED_STATUSES:
        return "failed", None
    return "processing", None


def _finalize(
    session: Session, assessment: Assessment, bag: AssessmentExecution
) -> None:
    from app.services.assessment.api.callbacks import deliver
    from app.services.assessment.api.result_files import finalize_result_files
    from app.services.assessment.api.results import build_result

    bag.stage = ApiStage.ASSESSMENT
    bag.stage_status = StageStatus.COMPLETED
    api.save_execution_state(session=session, assessment=assessment, state=bag)

    result = build_result(session=session, assessment=assessment, bag=bag)
    errors = sum(1 for item in result.items if item.error)
    if result.items and errors == len(result.items):
        status = AssessmentStatus.FAILED
    elif errors:
        status = AssessmentStatus.COMPLETED_WITH_ERRORS
    else:
        status = AssessmentStatus.COMPLETED

    api.update_status(session=session, assessment=assessment, status=status)
    logger.info(
        "[_finalize] Completed | assessment_id=%s | status=%s | items=%s | errors=%s",
        assessment.id,
        status,
        len(result.items),
        errors,
    )

    # Before the callback check: durability must not depend on a callback existing.
    finalize_result_files(session=session, assessment=assessment, bag=bag)

    if bag.callback_url:
        deliver(
            session=session,
            assessment=assessment,
            result=result,
            callback_url=bag.callback_url,
            request_metadata=bag.request_metadata,
            failure_message=None,
        )


def _fail(
    session: Session,
    assessment: Assessment,
    bag: AssessmentExecution,
    message: str,
) -> None:
    from app.services.assessment.api.callbacks import deliver
    from app.services.assessment.api.result_files import finalize_result_files
    from app.services.assessment.api.results import build_result

    bag.stage_status = StageStatus.FAILED
    bag.error = message
    api.save_execution_state(session=session, assessment=assessment, state=bag)
    api.update_status(
        session=session, assessment=assessment, status=AssessmentStatus.FAILED
    )
    logger.error(
        "[_fail] Assessment failed | assessment_id=%s | message=%s",
        assessment.id,
        message,
    )

    # Before the callback check: durability must not depend on a callback existing.
    finalize_result_files(
        session=session, assessment=assessment, bag=bag, failure_message=message
    )

    if bag.callback_url:
        result = build_result(session=session, assessment=assessment, bag=bag)
        deliver(
            session=session,
            assessment=assessment,
            result=result,
            callback_url=bag.callback_url,
            request_metadata=bag.request_metadata,
            failure_message=message,
        )


def _advance_or_finalize(
    *,
    session: Session,
    assessment: Assessment,
    blob: AssessmentConfigBlob,
    bag: AssessmentExecution,
    stage: ApiStage,
    organization_id: int,
    project_id: int,
) -> dict[str, bool]:
    """Move past a just-completed stage: submit the next one, or finalize if last.

    A stage whose row subset is empty (all rows gated out upstream) submits no batch
    (``_submit_stage`` returns False); it is treated as completed and skipped, recursing
    so a chain of empty stages still terminates at ``_finalize``.
    """
    from app.services.assessment.api.submission_store import SubmissionUnavailableError

    nxt = bag.next_stage(stage)
    if nxt is None:
        _finalize(session, assessment, bag)
        return {"requeue": False}
    try:
        bag.stage = nxt
        bag.stage_status = StageStatus.PENDING
        submitted = _submit_stage(
            session=session,
            assessment=assessment,
            blob=blob,
            bag=bag,
            stage=nxt,
            organization_id=organization_id,
            project_id=project_id,
        )
    except SubmissionUnavailableError as exc:
        # Storage blip, not a bad run: the stage is still PENDING, so retry the task.
        logger.warning(
            "[_advance_or_finalize] Submission unreadable, will retry | "
            "assessment_id=%s | stage=%s | %s",
            assessment.id,
            nxt,
            exc,
        )
        return {"requeue": True}
    except Exception as exc:
        _fail(session, assessment, bag, str(exc))
        return {"requeue": False}
    if not submitted:
        bag.stage_status = StageStatus.COMPLETED
        api.save_execution_state(session=session, assessment=assessment, state=bag)
        return _advance_or_finalize(
            session=session,
            assessment=assessment,
            blob=blob,
            bag=bag,
            stage=nxt,
            organization_id=organization_id,
            project_id=project_id,
        )
    return {"requeue": True}


def run_batch_stage(
    *, assessment_id: UUID, organization_id: int, project_id: int
) -> dict[str, bool]:
    """Run one step of the staged pipeline. Returns ``{"requeue": bool}``.

    Idempotent: keyed off ``stage_status`` in the bag — a redelivery either re-polls
    the in-flight batch or re-submits a stage that was never dispatched.
    """
    from app.services.assessment.api.result_files import record_stage_dump
    from app.services.assessment.api.submission_store import SubmissionUnavailableError

    with Session(engine) as session:
        assessment = session.exec(
            select(Assessment)
            .where(col(Assessment.id) == assessment_id)
            .with_for_update(skip_locked=True)
        ).first()
        if assessment is None:
            logger.warning(
                "[run_batch_stage] assessment_id=%s missing or held by another task",
                assessment_id,
            )
            return {"requeue": False}
        if assessment.status in AssessmentStatus.terminal():
            return {"requeue": False}

        bag = api.load_execution_state(assessment)
        if bag is None:
            logger.error(
                "[run_batch_stage] uninitialised execution | assessment_id=%s",
                assessment_id,
            )
            return {"requeue": False}
        stage = bag.stage
        stage_status = bag.stage_status

        # Entry log: distinguishes "the task never ran" from "it ran and did nothing".
        logger.info(
            "[run_batch_stage] Step | assessment_id=%s | stage=%s | stage_status=%s",
            assessment_id,
            stage,
            stage_status,
        )

        # Resolving the stored blob can raise (deleted config version -> 404, or an
        # invalid/old-shape blob). Route these to _fail so the client gets a terminal
        # callback instead of the run stranding in PROCESSING.
        try:
            blob = resolve_blob(session, assessment)
        except Exception as exc:
            _fail(session, assessment, bag, str(exc))
            return {"requeue": False}

        if stage_status == StageStatus.PENDING:
            try:
                submitted = _submit_stage(
                    session=session,
                    assessment=assessment,
                    blob=blob,
                    bag=bag,
                    stage=stage,
                    organization_id=organization_id,
                    project_id=project_id,
                )
            except SubmissionUnavailableError as exc:
                # Storage blip, not a bad run: the stage stays PENDING, retry the task.
                logger.warning(
                    "[run_batch_stage] Submission unreadable, will retry | "
                    "assessment_id=%s | stage=%s | %s",
                    assessment_id,
                    stage,
                    exc,
                )
                return {"requeue": True}
            except Exception as exc:
                # Credential/provider/network errors from _submit_provider_batch, not
                # just ValueError — all are terminal for this run.
                _fail(session, assessment, bag, str(exc))
                return {"requeue": False}
            if not submitted:
                # Empty subset (all rows gated out): the stage is done — advance/finalize
                # rather than re-submitting a PENDING stage forever.
                bag.stage_status = StageStatus.COMPLETED
                api.save_execution_state(
                    session=session, assessment=assessment, state=bag
                )
                return _advance_or_finalize(
                    session=session,
                    assessment=assessment,
                    blob=blob,
                    bag=bag,
                    stage=stage,
                    organization_id=organization_id,
                    project_id=project_id,
                )
            return {"requeue": True}

        # stage_status == PROCESSING: poll the in-flight batch.
        batch_id = bag.stage_batches.get(stage)
        batch_job = (
            get_batch_job(session=session, batch_job_id=batch_id) if batch_id else None
        )
        if batch_job is None:
            _fail(session, assessment, bag, f"Stage {stage} batch not found")
            return {"requeue": False}

        provider_name, _ = stage_provider_model(blob, stage)
        try:
            provider = _build_batch_provider(
                session=session,
                provider_name=provider_name,
                organization_id=organization_id,
                project_id=project_id,
            )
            outcome, results = _poll_outcome(
                session, provider, batch_job, assessment.id
            )
        except Exception as exc:
            # Transient (network/provider hiccup) — the batch is still running; retry.
            logger.warning(
                "[run_batch_stage] poll error, will retry | assessment_id=%s | stage=%s | %s",
                assessment_id,
                stage,
                exc,
            )
            return {"requeue": True}

        if outcome == "processing":
            return {"requeue": True}
        if outcome == "failed":
            _fail(
                session,
                assessment,
                bag,
                batch_job.error_message or f"Stage {stage} batch failed",
            )
            return {"requeue": False}

        # outcome == "completed"
        kind = bag.stage_kind(stage)
        parsed = parse_batch_results(results or [], provider_name)
        _record_stage(bag, stage, kind, parsed)
        bag.stage_status = StageStatus.COMPLETED
        bag.stage_errors[stage] = {
            idx: error for idx, out in parsed.items() if (error := out.get("error"))
        }

        # Per stage, not only at terminal time: a run that never terminates still has dumps.
        record_stage_dump(
            session=session,
            assessment=assessment,
            stage=stage,
            url=batch_job.raw_output_url,
        )
        api.save_execution_state(session=session, assessment=assessment, state=bag)

        return _advance_or_finalize(
            session=session,
            assessment=assessment,
            blob=blob,
            bag=bag,
            stage=stage,
            organization_id=organization_id,
            project_id=project_id,
        )
