"""CRUD operations for uploaded assessment submissions."""

import logging
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.util import now
from app.models.assessment import Assessment, AssessmentSubmission

logger = logging.getLogger(__name__)


def get_submission_by_name(
    *, session: Session, name: str, organization_id: int, project_id: int
) -> AssessmentSubmission | None:
    """Fetch a submission by the columns the unique constraint covers."""
    statement = (
        select(AssessmentSubmission)
        .where(AssessmentSubmission.name == name)
        .where(AssessmentSubmission.organization_id == organization_id)
        .where(AssessmentSubmission.project_id == project_id)
    )
    return session.exec(statement).first()


def create_submission(
    *,
    session: Session,
    name: str,
    object_store_url: str,
    total_items: int,
    organization_id: int,
    project_id: int,
    description: str | None = None,
) -> AssessmentSubmission:
    """Record an uploaded submission file."""
    submission = AssessmentSubmission(
        name=name,
        description=description,
        object_store_url=object_store_url,
        total_items=total_items,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )

    try:
        session.add(submission)
        session.commit()
        session.refresh(submission)
    except IntegrityError as e:
        # Backstop for two concurrent uploads; the caller already checks the name.
        session.rollback()
        logger.warning(
            "[create_submission] Name already exists | name=%s | org_id=%s | project_id=%s",
            name,
            organization_id,
            project_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"Submission with name '{name}' already exists in this "
                "organization and project. Please choose a different name."
            ),
        ) from e
    except Exception as e:
        session.rollback()
        logger.error(
            "[create_submission] Failed to create submission | name=%s",
            name,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to save the submission metadata."
        ) from e

    logger.info(
        "[create_submission] Created | id=%s | name=%s | rows=%s | org_id=%s | project_id=%s",
        submission.id,
        name,
        total_items,
        organization_id,
        project_id,
    )
    return submission


def get_submission_by_id(
    *,
    session: Session,
    submission_id: UUID,
    organization_id: int,
    project_id: int,
) -> AssessmentSubmission:
    """Fetch a submission by id, scoped to organization and project."""
    statement = (
        select(AssessmentSubmission)
        .where(AssessmentSubmission.id == submission_id)
        .where(AssessmentSubmission.organization_id == organization_id)
        .where(AssessmentSubmission.project_id == project_id)
    )
    submission = session.exec(statement).first()
    if not submission:
        raise HTTPException(
            status_code=404,
            detail=f"Submission {submission_id} not found or not accessible",
        )
    return submission


def list_submissions(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[AssessmentSubmission]:
    """List submissions for an organization and project, newest first."""
    statement = (
        select(AssessmentSubmission)
        .where(AssessmentSubmission.organization_id == organization_id)
        .where(AssessmentSubmission.project_id == project_id)
        .order_by(AssessmentSubmission.inserted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.exec(statement).all())


def delete_submission(
    *, session: Session, submission: AssessmentSubmission
) -> str | None:
    """Delete a submission no assessment references. Returns a reason when refused."""
    statement = select(Assessment).where(Assessment.submission_id == submission.id)
    assessments = session.exec(statement).all()
    if assessments:
        return (
            f"Cannot delete submission {submission.id}: it is being used by "
            f"{len(assessments)} assessment(s). Please delete the assessments first."
        )

    submission_id = submission.id
    submission_name = submission.name
    try:
        session.delete(submission)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(
            "[delete_submission] Failed to delete | submission_id=%s",
            submission.id,
            exc_info=True,
        )
        return f"Failed to delete submission: {e}"

    logger.info(
        "[delete_submission] Deleted | id=%s | name=%s", submission_id, submission_name
    )
    return None
