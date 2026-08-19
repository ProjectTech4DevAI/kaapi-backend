"""BATCH API-client staged pipeline: stage build/submit, verdict parsing, advance.

One config -> one execution -> an ordered series of provider batches:
all GATE pre-filters first (config order), then all PASS-THROUGH pre-filters,
then the assessment stage. Runtime state lives entirely in the execution bag
(``AssessmentRun.execution``); no dedicated columns.

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
from enum import StrEnum
from typing import Any, cast

from sqlmodel import Session

from app.core.batch import (
    BATCH_KEY,
    AnthropicBatchProvider,
    BatchJobState,
    MessageBatchStatus,
    OpenAIBatchProvider,
    extract_text_from_response_dict,
    get_gemini_batch_provider,
    is_vertex_batch_provider,
    poll_batch_status,
    process_completed_batch,
    start_batch_job,
)
from app.core.batch.base import BatchProvider
from app.core.config import settings
from app.core.db import engine
from app.crud.assessment import api
from app.crud.assessment.batch import (
    build_anthropic_jsonl,
    build_google_jsonl,
    build_openai_jsonl,
)
from app.services.assessment.utils.attachments import rewrite_gcs_attachment_urls
from app.crud.job import get_batch_job
from app.models.assessment import (
    Assessment,
    AssessmentAttachment,
    AssessmentRun,
    AssessmentStatus,
    BatchInput,
    BatchRunState,
    ParsedResult,
    StageStatus,
    Verdict,
)
from app.models.batch_job import BatchJob, BatchJobType
from app.models.config.assessment_blob import (
    AssessmentConfigBlob,
    AssessmentPreFilters,
    DuplicateDetectionFilter,
    TopicRelevanceFilter,
)
from app.models.llm.constants import DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS
from app.services.assessment.mappers import (
    map_kaapi_to_anthropic_params,
    map_kaapi_to_google_params,
    map_kaapi_to_openai_params,
)
from app.services.llm.providers.registry import LLMProvider
from app.utils import get_anthropic_client, get_openai_client

logger = logging.getLogger(__name__)

# Re-poll cadence for a stage's provider batch, mirroring the assessment cron tick.
POLL_COUNTDOWN_SECONDS = settings.CRON_INTERVAL_MINUTES * 60


class ApiStage(StrEnum):
    """Pipeline stage identifiers; names match the pre-filter config + result fields."""

    TOPIC_RELEVANCE = "topic_relevance"
    DUPLICATE_DETECTION = "duplicate_detection"
    ASSESSMENT = "assessment"


class StageKind(StrEnum):
    GATE = "GATE"
    PASS_THROUGH = "PASS_THROUGH"
    ASSESSMENT = "ASSESSMENT"


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
    LLMProvider.GOOGLE_GCP,
    LLMProvider.ANTHROPIC,
}


def is_supported_provider(provider_name: str) -> bool:
    return provider_name in _SUPPORTED_PROVIDERS


def build_pipeline(
    pre_filters: AssessmentPreFilters | None,
) -> list[dict[str, str]]:
    """Ordered stages: GATE pre-filters, then PASS-THROUGH pre-filters, then assessment."""
    present: list[tuple[ApiStage, TopicRelevanceFilter | DuplicateDetectionFilter]] = []
    if pre_filters is not None:
        if pre_filters.topic_relevance is not None:
            present.append((ApiStage.TOPIC_RELEVANCE, pre_filters.topic_relevance))
        if pre_filters.duplicate_detection is not None:
            present.append(
                (ApiStage.DUPLICATE_DETECTION, pre_filters.duplicate_detection)
            )

    gates = [
        {"stage": stage.value, "kind": StageKind.GATE.value}
        for stage, flt in present
        if flt.stop_on_fail
    ]
    passthrough = [
        {"stage": stage.value, "kind": StageKind.PASS_THROUGH.value}
        for stage, flt in present
        if not flt.stop_on_fail
    ]
    return [
        *gates,
        *passthrough,
        {"stage": ApiStage.ASSESSMENT.value, "kind": StageKind.ASSESSMENT.value},
    ]


def next_stage(pipeline: list[dict[str, str]], current: str) -> str | None:
    stages = [s["stage"] for s in pipeline]
    idx = stages.index(current)
    return stages[idx + 1] if idx + 1 < len(stages) else None


def _stage_kind(pipeline: list[dict[str, str]], stage: str) -> str:
    for entry in pipeline:
        if entry["stage"] == stage:
            return entry["kind"]
    raise ValueError(f"[_stage_kind] Stage {stage} not in pipeline")


def build_rows(
    batch_input: BatchInput,
    input_columns: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[str], list[AssessmentAttachment]]:
    """Split each submission into text cells + attachment specs for the JSONL builders.

    Column kinds come from the config's ``input_schema``: a column typed image/pdf is
    an attachment column (url-format only); any column not declared there is text. Cells are
    plain strings (an attachment cell holds the url), so the URL-only JSONL builders consume
    them unchanged.
    """
    submissions = batch_input.data
    input_columns = input_columns or {}
    columns = list(submissions[0].keys()) if submissions else []

    text_columns: list[str] = []
    attachments: list[AssessmentAttachment] = []
    for column in columns:
        spec = input_columns.get(column) or {}
        col_type = spec.get("type", "text")
        if col_type in ("image", "pdf"):
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

    rows = [{col: str(sub.get(col, "")) for col in columns} for sub in submissions]
    return rows, text_columns, attachments


def _stage_prompt(batch_input: BatchInput, stage: str) -> str | None:
    if stage == ApiStage.ASSESSMENT:
        return batch_input.query
    # Pre-filter criteria live in params.instructions (system prompt); the item's
    # own columns + attachments are the user content, so there is no per-row
    # prompt template for a pre-filter stage.
    return None


def _prefilter_for_stage(
    blob: AssessmentConfigBlob, stage: str
) -> TopicRelevanceFilter | DuplicateDetectionFilter:
    """The pre-filter config object for a pre-filter stage (topic_relevance / duplicate_detection)."""
    pre = blob.pre_filters
    if pre is not None:
        if stage == ApiStage.TOPIC_RELEVANCE and pre.topic_relevance is not None:
            return pre.topic_relevance
        if (
            stage == ApiStage.DUPLICATE_DETECTION
            and pre.duplicate_detection is not None
        ):
            return pre.duplicate_detection
    raise ValueError(
        f"[_prefilter_for_stage] No pre-filter configured for stage {stage}"
    )


def _stage_params(blob: AssessmentConfigBlob, stage: str) -> dict[str, Any]:
    """Kaapi params for a stage: the assessment uses its own params + output schema; each
    pre-filter uses ITS own params (criteria in params.instructions), with the verdict
    schema + gate directive layered on.
    """
    if stage == ApiStage.ASSESSMENT:
        params = dict(blob.assessment.params)
        json_schema = params.pop("json_output_schema", None)
        params.pop(
            "input_schema", None
        )  # request-validation only, not a provider param
        if json_schema is not None:
            params["output_schema"] = json_schema  # provider param name
        return params

    flt = _prefilter_for_stage(blob, stage)
    params = dict(flt.params)
    params["output_schema"] = PREFILTER_VERDICT_SCHEMA
    # The config's criteria (mandatory params.instructions) is the system prompt;
    # append the gate directive so the model returns the verdict+reasoning contract.
    criteria = params.get("instructions") or ""
    params["instructions"] = f"{criteria}\n\n{_PREFILTER_INSTRUCTION}"
    if isinstance(flt, DuplicateDetectionFilter) and flt.knowledge_base_id:
        params["knowledge_base_ids"] = [flt.knowledge_base_id]
    return params


def _stage_provider_model(blob: AssessmentConfigBlob, stage: str) -> tuple[str, str]:
    """Provider + model for a stage: each pre-filter uses its own; assessment uses the config's."""
    if stage in (ApiStage.TOPIC_RELEVANCE, ApiStage.DUPLICATE_DETECTION):
        flt = _prefilter_for_stage(blob, stage)
        return flt.provider, flt.params["model"]
    return blob.assessment.provider, blob.assessment.params["model"]


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
    elif provider_name in (LLMProvider.GOOGLE, LLMProvider.GOOGLE_GCP):
        mapped, _ = map_kaapi_to_google_params(params)
        jsonl = build_google_jsonl(
            rows, text_columns, attachments, prompt, mapped, row_indices
        )
        provider = get_gemini_batch_provider(
            session=session,
            organization_id=organization_id,
            project_id=project_id,
            provider_name=provider_name,
            model=model,
        )
        # Vertex takes a bare model id; AI-Studio uses the "models/" prefix.
        config = (
            {"display_name": description}
            if is_vertex_batch_provider(provider_name)
            else {"display_name": description, "model": f"models/{model}"}
        )
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
    if provider_name in (LLMProvider.GOOGLE, LLMProvider.GOOGLE_GCP):
        return get_gemini_batch_provider(
            session=session,
            organization_id=organization_id,
            project_id=project_id,
            provider_name=provider_name,
        )
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

    if provider_name in (LLMProvider.GOOGLE, LLMProvider.GOOGLE_GCP):
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


def _parse_verdict(output: str | None) -> Verdict:
    """Read a pre-filter verdict. Unparseable output fails open (verdict=True)."""
    if not output:
        return {"verdict": True, "reasoning": ""}
    try:
        data = json.loads(output)
        return {
            "verdict": bool(data.get("verdict", True)),
            "reasoning": str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning(
            "[_parse_verdict] Unparseable verdict, failing open | output=%s",
            output[:200],
        )
        return {"verdict": True, "reasoning": ""}


def _record_stage(
    bag: BatchRunState, stage: str, kind: str, parsed: dict[int, ParsedResult]
) -> None:
    """Fold a completed stage's parsed results into the bag per its kind."""
    if kind == StageKind.ASSESSMENT:
        return  # assessment outputs are read from object store at result time

    verdicts: dict[str, Verdict] = {}
    passed_count = 0
    for idx, out in parsed.items():
        verdict = _parse_verdict(out.get("output"))
        verdicts[str(idx)] = verdict
        if verdict["verdict"]:
            passed_count += 1
        elif kind == StageKind.GATE:
            bag["gate_passed"][idx] = False

    bag.setdefault("verdicts", {})[stage] = verdicts
    bag.setdefault("counters", {})[stage] = {
        "total": len(parsed),
        "passed": passed_count,
        "rejected": len(parsed) - passed_count,
    }


def _row_subset(bag: BatchRunState, stage: str, kind: str, total: int) -> list[int]:
    if kind == StageKind.ASSESSMENT:
        return [i for i in range(total) if bag["gate_passed"][i]]
    return list(range(total))


def _submit_stage(
    *,
    session: Session,
    execution: AssessmentRun,
    blob: AssessmentConfigBlob,
    batch_input: BatchInput,
    bag: BatchRunState,
    stage: str,
    organization_id: int,
    project_id: int,
) -> bool:
    """Build + submit the current stage's batch on its row subset. Returns success."""
    kind = _stage_kind(bag["pipeline"], stage)
    input_columns = blob.assessment.params.get("input_schema")
    rows, text_columns, attachments = build_rows(batch_input, input_columns)
    subset = _row_subset(bag, stage, kind, len(rows))

    if not subset:
        # No rows left for this stage (everything gated out upstream). Persist the
        # empty counters and return False so the caller advances/finalizes instead of
        # re-submitting a PENDING stage forever.
        logger.info(
            "[_submit_stage] Empty subset, skipping | execution_id=%s | stage=%s",
            execution.id,
            stage,
        )
        bag.setdefault("counters", {})[stage] = {"total": 0, "passed": 0, "rejected": 0}
        api.save_execution_state(session=session, execution=execution, state=bag)
        return False

    provider_name, model = _stage_provider_model(blob, stage)
    batch_job = _submit_provider_batch(
        session=session,
        provider_name=provider_name,
        model=model,
        rows=[rows[i] for i in subset],
        text_columns=text_columns,
        attachments=attachments,
        prompt=_stage_prompt(batch_input, stage),
        params=_stage_params(blob, stage),
        row_indices=subset,
        organization_id=organization_id,
        project_id=project_id,
        description=f"assessment-{execution.id}-{stage}",
    )

    bag["stage"] = stage
    bag["stage_status"] = StageStatus.PROCESSING.value
    bag.setdefault("stage_batches", {})[stage] = batch_job.id
    api.set_execution_batch_job(
        session=session, execution=execution, batch_job_id=batch_job.id
    )
    api.save_execution_state(session=session, execution=execution, state=bag)
    logger.info(
        "[_submit_stage] Submitted | execution_id=%s | stage=%s | batch_job=%s | rows=%s",
        execution.id,
        stage,
        batch_job.id,
        len(subset),
    )
    return True


def _poll_outcome(
    session: Session, provider: BatchProvider, batch_job: BatchJob
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
                session=session, provider=provider, batch_job=batch_job
            )
            return "completed", results
        return "processing", None  # output not ready yet
    if status in _FAILED_STATUSES:
        return "failed", None
    return "processing", None


def _finalize(
    session: Session,
    execution: AssessmentRun,
    assessment: Assessment,
    bag: BatchRunState,
) -> None:
    from app.services.assessment.api.callbacks import deliver
    from app.services.assessment.api.results import build_result

    bag["stage"] = ApiStage.ASSESSMENT.value
    bag["stage_status"] = StageStatus.COMPLETED.value
    api.save_execution_state(session=session, execution=execution, state=bag)

    result = build_result(session=session, assessment=assessment)
    errors = sum(1 for item in result.items if item.error)
    if result.items and errors == len(result.items):
        status = AssessmentStatus.FAILED
    elif errors:
        status = AssessmentStatus.COMPLETED_WITH_ERRORS
    else:
        status = AssessmentStatus.COMPLETED

    api.update_status(session=session, obj=execution, status=status)
    api.update_status(session=session, obj=assessment, status=status)
    logger.info(
        "[_finalize] Completed | execution_id=%s | status=%s | items=%s | errors=%s",
        execution.id,
        status,
        len(result.items),
        errors,
    )

    callback_url = bag.get("callback_url")
    if callback_url:
        deliver(
            assessment=assessment,
            result=result,
            callback_url=callback_url,
            request_metadata=bag.get("request_metadata"),
        )


def _fail(
    session: Session,
    execution: AssessmentRun,
    assessment: Assessment,
    bag: BatchRunState,
    message: str,
) -> None:
    from app.services.assessment.api.callbacks import deliver
    from app.services.assessment.api.results import build_result

    bag["stage_status"] = StageStatus.FAILED.value
    bag["error"] = message
    api.save_execution_state(session=session, execution=execution, state=bag)
    execution.error_message = message
    api.update_status(session=session, obj=execution, status=AssessmentStatus.FAILED)
    api.update_status(session=session, obj=assessment, status=AssessmentStatus.FAILED)
    logger.error(
        "[_fail] Execution failed | execution_id=%s | message=%s", execution.id, message
    )

    callback_url = bag.get("callback_url")
    if callback_url:
        result = build_result(session=session, assessment=assessment)
        deliver(
            assessment=assessment,
            result=result,
            callback_url=callback_url,
            request_metadata=bag.get("request_metadata"),
        )


def _advance_or_finalize(
    *,
    session: Session,
    execution: AssessmentRun,
    assessment: Assessment,
    blob: AssessmentConfigBlob,
    batch_input: BatchInput,
    bag: BatchRunState,
    stage: str,
    organization_id: int,
    project_id: int,
) -> dict[str, bool]:
    """Move past a just-completed stage: submit the next one, or finalize if last.

    A stage whose row subset is empty (all rows gated out upstream) submits no batch
    (``_submit_stage`` returns False); it is treated as completed and skipped, recursing
    so a chain of empty stages still terminates at ``_finalize``.
    """
    nxt = next_stage(bag["pipeline"], stage)
    if nxt is None:
        _finalize(session, execution, assessment, bag)
        return {"requeue": False}
    try:
        bag["stage"] = nxt
        bag["stage_status"] = StageStatus.PENDING.value
        submitted = _submit_stage(
            session=session,
            execution=execution,
            blob=blob,
            batch_input=batch_input,
            bag=bag,
            stage=nxt,
            organization_id=organization_id,
            project_id=project_id,
        )
    except Exception as exc:
        _fail(session, execution, assessment, bag, str(exc))
        return {"requeue": False}
    if not submitted:
        bag["stage_status"] = StageStatus.COMPLETED.value
        api.save_execution_state(session=session, execution=execution, state=bag)
        return _advance_or_finalize(
            session=session,
            execution=execution,
            assessment=assessment,
            blob=blob,
            batch_input=batch_input,
            bag=bag,
            stage=nxt,
            organization_id=organization_id,
            project_id=project_id,
        )
    return {"requeue": True}


def run_batch_stage(
    *, execution_id: int, organization_id: int, project_id: int
) -> dict[str, bool]:
    """Drive one tick of the staged pipeline. Returns ``{"requeue": bool}``.

    Idempotent: keyed off ``stage_status`` in the bag — a redelivery either re-polls
    the in-flight batch or re-submits a stage that was never dispatched.
    """
    with Session(engine) as session:
        execution = session.get(AssessmentRun, execution_id)
        if execution is None:
            logger.error("[run_batch_stage] execution_id=%s not found", execution_id)
            return {"requeue": False}
        assessment = session.get(Assessment, execution.assessment_id)
        if assessment is None:
            logger.error(
                "[run_batch_stage] parent assessment missing | execution_id=%s",
                execution_id,
            )
            return {"requeue": False}
        if execution.status in (
            AssessmentStatus.COMPLETED,
            AssessmentStatus.COMPLETED_WITH_ERRORS,
            AssessmentStatus.FAILED,
        ):
            return {"requeue": False}

        bag = cast(BatchRunState, dict(execution.execution or {}))
        stage = bag.get("stage")
        stage_status = bag.get("stage_status")
        if not stage or not bag.get("pipeline"):
            logger.error(
                "[run_batch_stage] uninitialised bag | execution_id=%s", execution_id
            )
            return {"requeue": False}

        # Resolving the stored blob can raise (deleted config version -> 404, or an
        # invalid/old-shape blob). Route these to _fail so the client gets a terminal
        # callback instead of the run stranding in PROCESSING.
        try:
            batch_input = BatchInput.model_validate(assessment.input)
            blob = AssessmentConfigBlob.model_validate(
                _resolve_blob(session, execution, project_id)
            )
        except Exception as exc:
            _fail(session, execution, assessment, bag, str(exc))
            return {"requeue": False}

        if stage_status == StageStatus.PENDING:
            try:
                submitted = _submit_stage(
                    session=session,
                    execution=execution,
                    blob=blob,
                    batch_input=batch_input,
                    bag=bag,
                    stage=stage,
                    organization_id=organization_id,
                    project_id=project_id,
                )
            except Exception as exc:
                # Credential/provider/network errors from _submit_provider_batch, not
                # just ValueError — all are terminal for this run.
                _fail(session, execution, assessment, bag, str(exc))
                return {"requeue": False}
            if not submitted:
                # Empty subset (all rows gated out): the stage is done — advance/finalize
                # rather than re-submitting a PENDING stage forever.
                bag["stage_status"] = StageStatus.COMPLETED.value
                api.save_execution_state(
                    session=session, execution=execution, state=bag
                )
                return _advance_or_finalize(
                    session=session,
                    execution=execution,
                    assessment=assessment,
                    blob=blob,
                    batch_input=batch_input,
                    bag=bag,
                    stage=stage,
                    organization_id=organization_id,
                    project_id=project_id,
                )
            return {"requeue": True}

        # stage_status == PROCESSING: poll the in-flight batch.
        batch_id = (bag.get("stage_batches") or {}).get(stage)
        batch_job = (
            get_batch_job(session=session, batch_job_id=batch_id) if batch_id else None
        )
        if batch_job is None:
            _fail(session, execution, assessment, bag, f"Stage {stage} batch not found")
            return {"requeue": False}

        try:
            provider = _build_batch_provider(
                session=session,
                provider_name=bag["provider"],
                organization_id=organization_id,
                project_id=project_id,
            )
            outcome, results = _poll_outcome(session, provider, batch_job)
        except Exception as exc:
            # Transient (network/provider hiccup) — the batch is still running; retry.
            logger.warning(
                "[run_batch_stage] poll error, will retry | execution_id=%s | stage=%s | %s",
                execution_id,
                stage,
                exc,
            )
            return {"requeue": True}

        if outcome == "processing":
            return {"requeue": True}
        if outcome == "failed":
            _fail(
                session,
                execution,
                assessment,
                bag,
                batch_job.error_message or f"Stage {stage} batch failed",
            )
            return {"requeue": False}

        # outcome == "completed"
        kind = _stage_kind(bag["pipeline"], stage)
        parsed = parse_batch_results(results or [], bag["provider"])
        _record_stage(bag, stage, kind, parsed)
        bag["stage_status"] = StageStatus.COMPLETED.value
        bag.setdefault("stage_output_urls", {})[stage] = batch_job.raw_output_url
        api.save_execution_state(session=session, execution=execution, state=bag)

        return _advance_or_finalize(
            session=session,
            execution=execution,
            assessment=assessment,
            blob=blob,
            batch_input=batch_input,
            bag=bag,
            stage=stage,
            organization_id=organization_id,
            project_id=project_id,
        )


def _resolve_blob(
    session: Session, execution: AssessmentRun, project_id: int
) -> dict[str, Any]:
    """Fetch the execution's stored ASSESSMENT config_blob dict for re-parsing."""
    from app.crud.config.version import ConfigVersionCrud
    from app.models.config.config import ConfigTag

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=execution.config_id,
        project_id=project_id,
        tag=ConfigTag.ASSESSMENT,
    )
    version = version_crud.exists_or_raise(version_number=execution.config_version)
    return version.config_blob
