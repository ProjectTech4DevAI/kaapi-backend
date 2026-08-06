"""Stage registry, pipeline ordering, and Batch API executor."""

import logging
from collections.abc import Callable
from typing import Any

from sqlmodel import Session

from app.core.batch import (
    AnthropicBatchProvider,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    download_batch_results,
    start_batch_job,
)
from app.core.batch.base import BatchProvider
from app.core.batch.client import GeminiClient
from app.core.cloud import get_cloud_storage
from app.crud.assessment.core import _read_exec, _write_exec
from app.models.assessment import AssessmentRun, AssessmentStatus, Stage, StageStatus
from app.models.batch_job import BatchJob, BatchJobType
from app.services.assessment.prefilter import constants, resolve_prefilter_settings
from app.services.assessment.prefilter.duplicate_detection import (
    build_duplicate_detection_requests,
    parse_duplicate_detection_results,
)
from app.services.assessment.prefilter.topic_relevance import (
    build_topic_relevance_requests,
    parse_topic_relevance_results,
)
from app.services.llm.providers.registry import LLMProvider
from app.utils import get_anthropic_client, get_openai_client

logger = logging.getLogger(__name__)

# Stages that gate the pipeline (only ACCEPTed rows continue). Others annotate.
GATE_STAGES = {Stage.PRE_FILTER_TOPIC_RELEVANCE}

# Result parser per stage: raw batch results -> {row_id: result dict}.
STAGE_PARSERS: dict[str, Callable[[list[dict]], dict[int, dict[str, Any]]]] = {
    Stage.PRE_FILTER_TOPIC_RELEVANCE: parse_topic_relevance_results,
    Stage.PRE_FILTER_DUPLICATE_DETECTION: parse_duplicate_detection_results,
}


def build_pipeline(assessment_input: dict[str, Any]) -> dict[str, Any]:
    """Build the ordered stage config; prefilter stages added only when configured."""
    cfg = resolve_prefilter_settings(assessment_input.get("prefilter_config") or {})
    stages: list[dict[str, Any]] = []
    if cfg["tr_enabled"]:
        stages.append({"stage": Stage.PRE_FILTER_TOPIC_RELEVANCE, "type": "GO_NO_GO"})
    if cfg["dup_enabled"]:
        stages.append(
            {"stage": Stage.PRE_FILTER_DUPLICATE_DETECTION, "type": "ANNOTATIVE"}
        )
    stages.append({"stage": Stage.L2_ASSESSMENT, "type": "ASSESSMENT"})

    for order, entry in enumerate(stages, start=1):
        entry["order"] = order
    return {"stages": stages}


def ordered_stages(pipeline: dict[str, Any] | None) -> list[str]:
    """The stage names in execution order."""
    return [s["stage"] for s in (pipeline or {}).get("stages", [])]


def next_stage(
    pipeline: dict[str, Any] | None, current: str | None = None
) -> str | None:
    """First stage when ``current`` is None, else the stage after it (None if last)."""
    stages = ordered_stages(pipeline)
    if current is None:
        return stages[0] if stages else None
    if current in stages and stages.index(current) + 1 < len(stages):
        return stages[stages.index(current) + 1]
    return None


def submit_prefilter_batch(
    session: Session,
    organization_id: int,
    project_id: int,
    jsonl_data: list[dict[str, Any]],
    display_name: str,
) -> BatchJob:
    """Submit a prefilter batch on the configured provider and return the BatchJob."""
    base = constants.ASSESSMENT_PREFILTER_PROVIDER
    provider = _get_batch_provider(
        session=session,
        provider_name=base,
        organization_id=organization_id,
        project_id=project_id,
    )
    if base == "openai":
        config = {
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "description": display_name,
        }
    else:
        config = {
            "display_name": display_name,
            "model": f"models/{constants.ASSESSMENT_PREFILTER_MODEL}",
        }
    return start_batch_job(
        session=session,
        provider=provider,
        provider_name=base,
        job_type=BatchJobType.ASSESSMENT,
        organization_id=organization_id,
        project_id=project_id,
        jsonl_data=jsonl_data,
        config=config,
    )


def build_prefilter_requests(
    stage: str,
    rows: list[tuple[int, dict[str, str]]],
    cfg: dict[str, Any],
    attachments: list | None = None,
) -> list[dict[str, Any]]:
    """Build the JSONL request lines for a prefilter stage."""
    if stage == Stage.PRE_FILTER_TOPIC_RELEVANCE:
        return build_topic_relevance_requests(
            rows, cfg["tr_columns"], cfg["tr_prompt"], attachments
        )
    if stage == Stage.PRE_FILTER_DUPLICATE_DETECTION:
        return build_duplicate_detection_requests(rows, cfg["dup_columns"])
    raise ValueError(f"Unknown prefilter stage: {stage}")


def _get_batch_provider(
    session: Session,
    provider_name: str,
    organization_id: int,
    project_id: int,
) -> BatchProvider:
    """Build the batch provider instance for a given provider name."""
    if provider_name in (LLMProvider.OPENAI, LLMProvider.OPENAI_NATIVE):
        return OpenAIBatchProvider(
            client=get_openai_client(
                session=session, org_id=organization_id, project_id=project_id
            )
        )
    if provider_name in (
        LLMProvider.GOOGLE_AISTUDIO,
        LLMProvider.GOOGLE_AISTUDIO_NATIVE,
    ):
        gemini_client = GeminiClient.from_credentials(
            session=session, org_id=organization_id, project_id=project_id
        )
        return GeminiBatchProvider(client=gemini_client.client)
    if provider_name in (LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC_NATIVE):
        return AnthropicBatchProvider(
            client=get_anthropic_client(
                session=session, org_id=organization_id, project_id=project_id
            )
        )
    raise ValueError(f"Unsupported batch provider: {provider_name}")


def load_raw_batch_results(
    session: Session, batch_job: BatchJob, project_id: int
) -> list[dict[str, Any]]:
    """Load a completed batch's raw result lines (object store first, else provider)."""
    # Lazy import: app.services.assessment.utils.__init__ pulls in export, which
    # imports this module's package — a top-level import would be circular.
    from app.services.assessment.utils.parsing import parse_stored_results

    if batch_job.raw_output_url:
        try:
            storage = get_cloud_storage(session, project_id=project_id)
            raw = parse_stored_results(
                storage.stream(batch_job.raw_output_url).read().decode("utf-8")
            )
            if raw:
                return raw
        except Exception as exc:
            logger.warning(
                "[load_raw_batch_results] S3 read failed batch %s — %s",
                batch_job.id,
                exc,
            )
    provider = _get_batch_provider(
        session=session,
        provider_name=batch_job.provider,
        organization_id=batch_job.organization_id,
        project_id=project_id,
    )
    return download_batch_results(provider=provider, batch_job=batch_job)


def advance_or_finalize(run: AssessmentRun) -> str | None:
    """Advance the run to the next stage (returned) or finalize it (returns None)."""
    exec_bag = _read_exec(run)
    nxt = next_stage(exec_bag.get("pipeline"), exec_bag.get("stage"))
    if nxt:
        _write_exec(run, stage=nxt, stage_status=StageStatus.PENDING)
        return nxt
    _write_exec(run, stage=Stage.COMPLETED, stage_status=StageStatus.COMPLETED)
    run.status = AssessmentStatus.COMPLETED
    return None
