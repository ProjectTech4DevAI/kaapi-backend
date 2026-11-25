"""Duplicate-detection CRUD: one transaction per step of a job."""

from uuid import UUID

from pydantic import JsonValue
from sqlmodel import Session, col, select, update

from app.core.util import now
from app.models.assessment import (
    Assessment,
    AssessmentMethod,
    AssessmentRun,
    AssessmentStatus,
)
from app.models.assessment.duplicate_detection import (
    DetectionLevel,
    DuplicateDetectionCreate,
)


def create_assessment(
    *,
    session: Session,
    request: DuplicateDetectionCreate,
    organization_id: int,
    project_id: int,
) -> tuple[Assessment, list[AssessmentRun]]:
    """Insert the job and its strict and lenient runs in one commit."""
    assessment = Assessment(
        method=AssessmentMethod.DUPLICATE_DETECTION,
        input=request.model_dump(mode="json", exclude={"config"}),
        organization_id=organization_id,
        project_id=project_id,
    )

    runs = [
        AssessmentRun(
            assessment_id=assessment.id,
            config_id=getattr(request.config, level).id,
            config_version=getattr(request.config, level).version,
            total_items=1,
            execution={"level": level},
        )
        for level in DetectionLevel
    ]

    session.add(assessment)
    session.add_all(runs)
    session.commit()

    return assessment, runs


def get_assessment(
    *, session: Session, assessment_id: UUID, organization_id: int, project_id: int
) -> Assessment | None:
    """A duplicate-detection job owned by this project, or None."""
    return session.exec(
        select(Assessment).where(
            col(Assessment.id) == assessment_id,
            col(Assessment.method) == AssessmentMethod.DUPLICATE_DETECTION,
            col(Assessment.organization_id) == organization_id,
            col(Assessment.project_id) == project_id,
        )
    ).first()


def claim_assessment(
    *, session: Session, assessment_id: UUID, statuses: tuple[AssessmentStatus, ...]
) -> Assessment | None:
    """Take a job in one of `statuses` and mark it PROCESSING; None when it is in none of them."""
    claimed = session.exec(
        update(Assessment)
        .where(
            col(Assessment.id) == assessment_id,
            col(Assessment.status).in_(statuses),
        )
        .values(status=AssessmentStatus.PROCESSING, updated_at=now())
    )
    session.commit()

    if claimed.rowcount == 0:
        return None

    return session.get(Assessment, assessment_id)


def save_result(
    *,
    session: Session,
    assessment: Assessment,
    status: AssessmentStatus,
    outcomes: list[tuple[AssessmentRun, JsonValue, str | None]],
) -> None:
    """Write each run's result or error and the job's final status in one commit."""
    for run, result, error in outcomes:
        if error is None:
            run.execution = {**(run.execution or {}), "result": result}

        run.error_message = error
        run.status = AssessmentStatus.FAILED if error else AssessmentStatus.COMPLETED
        run.updated_at = now()
        session.add(run)

    assessment.status = status
    assessment.updated_at = now()
    session.add(assessment)

    session.commit()
