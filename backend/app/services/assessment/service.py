"""Assessment run orchestration service."""

import logging
from uuid import UUID

from asgi_correlation_id import correlation_id
from fastapi import HTTPException
from sqlmodel import Session

from app.crud.assessment import (
    create_assessment,
    create_assessment_run,
    get_assessment_dataset_by_id,
    get_assessment_runs_for_assessment,
    recompute_assessment_status,
)
from app.crud.config import ConfigCrud
from app.crud.evaluations.core import resolve_evaluation_config
from app.models.assessment import (
    Assessment,
    AssessmentAttachment,
    AssessmentConfigRef,
    AssessmentCreate,
    AssessmentResponse,
    AssessmentRun,
    AssessmentRunSummary,
    StageStatus,
)
from app.models.config.config import ConfigTag
from app.services.llm.providers.registry import LLMProvider

logger = logging.getLogger(__name__)

_SUPPORTED_BATCH_PROVIDERS = {
    LLMProvider.OPENAI,
    LLMProvider.OPENAI_NATIVE,
    LLMProvider.GOOGLE_AISTUDIO,
    LLMProvider.GOOGLE_AISTUDIO_NATIVE,
    LLMProvider.GOOGLE,
    LLMProvider.GOOGLE_NATIVE,
    LLMProvider.ANTHROPIC,
    LLMProvider.ANTHROPIC_NATIVE,
}


def _build_retry_request(
    *,
    experiment_name: str,
    dataset_id: int,
    runs: list[AssessmentRun],
) -> AssessmentCreate:
    if not runs:
        raise HTTPException(status_code=400, detail="No assessment runs found to retry")

    first_run = runs[0]
    assessment_input = first_run.input
    if not isinstance(assessment_input, dict):
        raise HTTPException(
            status_code=400,
            detail="Assessment input configuration is missing for retry",
        )

    attachments = assessment_input.get("attachments") or []
    configs: list[AssessmentConfigRef] = []
    for run in runs:
        if not run.config_id or run.config_version is None:
            raise HTTPException(
                status_code=400,
                detail=f"Config reference is missing for run {run.id}",
            )
        configs.append(
            AssessmentConfigRef(
                config_id=UUID(str(run.config_id)),
                config_version=run.config_version,
            )
        )

    return AssessmentCreate(
        experiment_name=experiment_name,
        dataset_id=dataset_id,
        prompt_template=assessment_input.get("prompt_template"),
        system_instruction=assessment_input.get("system_instruction"),
        text_columns=list(assessment_input.get("text_columns") or []),
        attachments=[AssessmentAttachment.model_validate(item) for item in attachments],
        output_schema=assessment_input.get("output_schema"),
        configs=configs,
        prefilter_config=assessment_input.get("prefilter_config"),
        post_processing_config=assessment_input.get("post_processing_config"),
    )


def start_assessment(
    session: Session,
    request: AssessmentCreate,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Validate, create Assessment + AssessmentRun records, dispatch Celery tasks.

    Each run is created with status='pending' and handed off to a Celery worker
    that runs prefilter filtering then submits the L2 batch.
    """
    from app.celery.tasks.job_execution import run_assessment_pipeline

    logger.info(
        "[start_assessment] Starting | experiment=%s | dataset_id=%s | configs=%s | org_id=%s",
        request.experiment_name,
        request.dataset_id,
        len(request.configs),
        organization_id,
    )

    dataset = get_assessment_dataset_by_id(
        session=session,
        dataset_id=request.dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    assessment_input: dict = {
        "prompt_template": request.prompt_template,
        "system_instruction": request.system_instruction,
        "text_columns": request.text_columns,
        "attachments": [att.model_dump() for att in request.attachments],
    }
    if request.output_schema:
        assessment_input["output_schema"] = request.output_schema
    if request.prefilter_config:
        assessment_input["prefilter_config"] = request.prefilter_config
    if request.post_processing_config:
        assessment_input["post_processing_config"] = request.post_processing_config

    config_crud = ConfigCrud(session=session, project_id=project_id)

    resolved_configs = []
    for cfg in request.configs:
        parent_config = config_crud.read_one(cfg.config_id)
        if parent_config is not None and parent_config.tag != ConfigTag.ASSESSMENT:
            tag_value = (
                parent_config.tag.value
                if parent_config.tag is not None
                else ConfigTag.DEFAULT.value
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Config {cfg.config_id} has tag '{tag_value}' "
                    f"and cannot be used for assessment. "
                    f"Only configs tagged 'ASSESSMENT' are allowed."
                ),
            )

        config_blob, error = resolve_evaluation_config(
            session=session,
            config_id=cfg.config_id,
            config_version=cfg.config_version,
            project_id=project_id,
            tag=ConfigTag.ASSESSMENT,
        )
        if error or config_blob is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Failed to resolve config {cfg.config_id} "
                    f"v{cfg.config_version}: {error}"
                ),
            )
        provider = config_blob.completion.provider or LLMProvider.OPENAI
        if provider not in _SUPPORTED_BATCH_PROVIDERS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Config {cfg.config_id} v{cfg.config_version} uses provider "
                    f"'{provider}', which is not supported for batch assessment. "
                    f"Supported providers: {sorted(_SUPPORTED_BATCH_PROVIDERS)}"
                ),
            )
        resolved_configs.append(cfg)

    assessment = create_assessment(
        session=session,
        experiment_name=request.experiment_name,
        dataset_id=request.dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    runs: list[AssessmentRun] = []
    trace_id = correlation_id.get() or ""

    for cfg in resolved_configs:
        run = create_assessment_run(
            session=session,
            assessment_id=assessment.id,
            config_id=cfg.config_id,
            config_version=cfg.config_version,
            assessment_input=assessment_input,
        )
        runs.append(run)

        run_assessment_pipeline.delay(
            run_id=run.id,
            organization_id=organization_id,
            project_id=project_id,
            trace_id=trace_id,
        )

        logger.info(
            "[start_assessment] Dispatched Celery task | run_id=%s | config_id=%s",
            run.id,
            cfg.config_id,
        )

    recompute_assessment_status(session=session, assessment_id=assessment.id)

    logger.info(
        "[start_assessment] Created assessment %s with %s runs | run_ids=%s",
        assessment.id,
        len(runs),
        [run.id for run in runs],
    )

    return AssessmentResponse(
        assessment_id=assessment.id,
        experiment_name=request.experiment_name,
        dataset_id=request.dataset_id,
        dataset_name=dataset.name,
        num_configs=len(runs),
        runs=[
            AssessmentRunSummary(
                run_id=run.id,
                assessment_id=run.assessment_id,
                config_id=str(run.config_id),
                config_version=run.config_version,
                status=run.status,
            )
            for run in runs
        ],
    )


def retry_assessment(
    session: Session,
    assessment: Assessment,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Create a new assessment using the same parent assessment inputs."""
    runs = get_assessment_runs_for_assessment(
        session=session, assessment_id=assessment.id
    )
    request = _build_retry_request(
        experiment_name=assessment.experiment_name,
        dataset_id=assessment.dataset_id,
        runs=runs,
    )
    return start_assessment(
        session=session,
        request=request,
        organization_id=organization_id,
        project_id=project_id,
    )


def retry_assessment_run(
    session: Session,
    run: AssessmentRun,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Create a new assessment using the same inputs as a single child run."""
    parent = getattr(run, "assessment", None) or session.get(
        Assessment, run.assessment_id
    )
    if not parent:
        raise HTTPException(
            status_code=404,
            detail=f"Parent assessment {run.assessment_id} not found",
        )
    request = _build_retry_request(
        experiment_name=parent.experiment_name,
        dataset_id=parent.dataset_id,
        runs=[run],
    )
    return start_assessment(
        session=session,
        request=request,
        organization_id=organization_id,
        project_id=project_id,
    )


def resume_assessment_run(
    session: Session,
    run: AssessmentRun,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Re-run a failed run from its failed stage, reusing completed upstream batches."""
    from app.celery.tasks.job_execution import run_assessment_pipeline
    from app.services.assessment.stages import ordered_stages

    if run.stage_status != StageStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run.id} is not in a failed state and cannot be resumed",
        )
    if run.stage not in ordered_stages(run.pipeline):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run.id} has no resumable failed stage",
        )

    parent = getattr(run, "assessment", None) or session.get(
        Assessment, run.assessment_id
    )
    if not parent:
        raise HTTPException(
            status_code=404,
            detail=f"Parent assessment {run.assessment_id} not found",
        )
    dataset = get_assessment_dataset_by_id(
        session=session,
        dataset_id=parent.dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    run.stage_status = StageStatus.PENDING
    run.status = "processing"
    run.error_message = None
    session.add(run)
    session.commit()
    session.refresh(run)
    recompute_assessment_status(session=session, assessment_id=run.assessment_id)

    logger.info(
        "[resume_assessment_run] Resuming run_id=%s from stage=%s",
        run.id,
        run.stage,
    )
    run_assessment_pipeline.delay(
        run_id=run.id,
        organization_id=organization_id,
        project_id=project_id,
        trace_id=correlation_id.get() or "",
    )

    return AssessmentResponse(
        assessment_id=parent.id,
        experiment_name=parent.experiment_name,
        dataset_id=parent.dataset_id,
        dataset_name=dataset.name if dataset else None,
        num_configs=1,
        runs=[
            AssessmentRunSummary(
                run_id=run.id,
                assessment_id=run.assessment_id,
                config_id=str(run.config_id),
                config_version=run.config_version,
                status=run.status,
            )
        ],
    )
