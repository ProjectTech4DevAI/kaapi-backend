"""Assessment evaluation orchestration service."""

import logging
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.assessment.batch import _resolve_config, submit_assessment_batch
from app.assessment.crud import (
    create_assessment,
    create_assessment_run,
    get_assessment_runs_for_manager,
    recompute_assessment_status,
    update_assessment_run_status,
)
from app.assessment.models import (
    AssessmentAttachment,
    AssessmentConfigRef,
    AssessmentCreate,
    AssessmentResponse,
    AssessmentRunSummary,
)
from app.crud.evaluations import get_dataset_by_id
from app.assessment.models import Assessment
from app.models.evaluation import EvaluationRun

logger = logging.getLogger(__name__)


def _build_retry_request(
    *,
    experiment_name: str,
    dataset_id: int,
    runs: list[EvaluationRun],
) -> AssessmentCreate:
    if not runs:
        raise HTTPException(status_code=400, detail="No assessment runs found to retry")

    first_run = runs[0]
    assessment_config = (first_run.score or {}).get("assessment_config")
    if not isinstance(assessment_config, dict):
        raise HTTPException(
            status_code=400,
            detail="Assessment configuration is missing for retry",
        )

    attachments = assessment_config.get("attachments") or []
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
        prompt_template=assessment_config.get("prompt_template"),
        text_columns=list(assessment_config.get("text_columns") or []),
        attachments=[AssessmentAttachment.model_validate(item) for item in attachments],
        output_schema=assessment_config.get("output_schema"),
        configs=configs,
    )


def start_assessment(
    session: Session,
    request: AssessmentCreate,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Start an assessment evaluation.

    Validates the dataset, resolves each config, creates one EvaluationRun per config,
    and kicks off batch processing for each.

    Args:
        session: Database session
        request: Validated request body
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        AssessmentResponse with created run summaries

    Raises:
        HTTPException: If dataset not found or configs invalid
    """
    logger.info(
        f"[start_assessment] Starting | "
        f"experiment={request.experiment_name} | "
        f"dataset_id={request.dataset_id} | "
        f"configs={len(request.configs)} | "
        f"org_id={organization_id}"
    )

    # 1. Validate dataset
    dataset = get_dataset_by_id(
        session=session,
        dataset_id=request.dataset_id,
        organization_id=organization_id,
        project_id=project_id,
    )
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {request.dataset_id} not found or not accessible",
        )

    # 2. Build assessment-specific config to store with each run
    assessment_config: dict[str, Any] = {
        "prompt_template": request.prompt_template,
        "text_columns": request.text_columns,
        "attachments": [a.model_dump() for a in request.attachments],
    }
    if request.output_schema:
        assessment_config["output_schema"] = request.output_schema

    # 3. Validate all configs first before creating any runs
    resolved_configs = []
    for cfg in request.configs:
        config_blob, error = _resolve_config(
            session=session,
            config_id=cfg.config_id,
            config_version=cfg.config_version,
            project_id=project_id,
        )
        if error or config_blob is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to resolve config {cfg.config_id} v{cfg.config_version}: {error}",
            )
        resolved_configs.append((cfg, config_blob))

    # 4. Create parent assessment manager row
    assessment = create_assessment(
        session=session,
        experiment_name=request.experiment_name,
        dataset_id=request.dataset_id,
        dataset_name=dataset.name,
        organization_id=organization_id,
        project_id=project_id,
        total_runs=len(resolved_configs),
    )

    # 5. Create one EvaluationRun per config and submit batches
    runs: list[EvaluationRun] = []
    for cfg, config_blob in resolved_configs:
        run = create_assessment_run(
            session=session,
            run_name=request.experiment_name,
            dataset_name=dataset.name,
            dataset_id=request.dataset_id,
            assessment_id=assessment.id,
            config_id=cfg.config_id,
            config_version=cfg.config_version,
            organization_id=organization_id,
            project_id=project_id,
            assessment_config=assessment_config,
        )

        # Submit batch for this run
        try:
            batch_job = submit_assessment_batch(
                session=session,
                eval_run=run,
                dataset=dataset,
                config_blob=config_blob,
                assessment_config=assessment_config,
                organization_id=organization_id,
                project_id=project_id,
            )

            run = update_assessment_run_status(
                session=session,
                eval_run=run,
                status="processing",
                batch_job_id=batch_job.id,
                total_items=batch_job.total_items,
            )
            recompute_assessment_status(session=session, assessment_id=assessment.id)

        except Exception as e:
            logger.error(
                f"[start_assessment] Failed to submit batch for run {run.id}: {e}",
                exc_info=True,
            )
            run = update_assessment_run_status(
                session=session,
                eval_run=run,
                status="failed",
                error_message=f"Batch submission failed: {str(e)}",
            )
            recompute_assessment_status(session=session, assessment_id=assessment.id)

        runs.append(run)

    recompute_assessment_status(session=session, assessment_id=assessment.id)

    logger.info(
        f"[start_assessment] Created assessment {assessment.id} with {len(runs)} runs | "
        f"run_ids={[r.id for r in runs]}"
    )

    return AssessmentResponse(
        assessment_id=assessment.id,
        experiment_name=request.experiment_name,
        dataset_id=request.dataset_id,
        dataset_name=dataset.name,
        num_configs=len(runs),
        runs=[
            AssessmentRunSummary(
                run_id=r.id,
                assessment_id=r.assessment_id,
                config_id=str(r.config_id),
                config_version=r.config_version,
                status=r.status,
            )
            for r in runs
        ],
    )


def retry_assessment(
    session: Session,
    assessment: Assessment,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Create a new assessment run using the same parent assessment inputs."""
    runs = get_assessment_runs_for_manager(session=session, assessment=assessment)
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
    run: EvaluationRun,
    organization_id: int,
    project_id: int,
) -> AssessmentResponse:
    """Create a new assessment using the same inputs as a single child run."""
    request = _build_retry_request(
        experiment_name=run.run_name,
        dataset_id=run.dataset_id,
        runs=[run],
    )
    return start_assessment(
        session=session,
        request=request,
        organization_id=organization_id,
        project_id=project_id,
    )
