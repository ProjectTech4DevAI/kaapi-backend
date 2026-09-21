"""Assessment submission-file endpoints."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core.cloud import get_cloud_storage
from app.crud.assessment.submission import (
    delete_submission,
    get_submission_by_id,
    list_submissions,
)
from app.models.assessment import (
    AssessmentSubmission,
    AssessmentSubmissionPreview,
    AssessmentSubmissionResponse,
)
from app.services.assessment.submission import preview_submission, upload_submission
from app.services.assessment.validators import validate_dataset_file
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter()


def _submission_to_response(
    submission: AssessmentSubmission,
    signed_url: str | None = None,
    preview: AssessmentSubmissionPreview | None = None,
) -> AssessmentSubmissionResponse:
    return AssessmentSubmissionResponse(
        submission_id=submission.id,
        name=submission.name,
        description=submission.description,
        total_items=submission.total_items,
        object_store_url=submission.object_store_url,
        signed_url=signed_url,
        preview=preview,
    )


@router.post(
    "/datasets",
    description=load_description("assessment/upload_dataset.md"),
    response_model=APIResponse[AssessmentSubmissionResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def upload_dataset(
    session: SessionDep,
    auth_context: AuthContextDep,
    file: UploadFile = File(..., description="CSV or Excel file to upload"),
    dataset_name: str = Form(..., description="Name for the submission"),
    description: str | None = Form(None, description="Optional description"),
) -> APIResponse[AssessmentSubmissionResponse]:
    """Upload a submission file (any CSV/Excel file, no column requirements)."""
    file_content, file_ext = await validate_dataset_file(file)

    submission = upload_submission(
        session=session,
        file_content=file_content,
        file_ext=file_ext,
        submission_name=dataset_name,
        description=description,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=_submission_to_response(submission))


@router.get(
    "/datasets",
    description=load_description("assessment/list_datasets.md"),
    response_model=APIResponse[list[AssessmentSubmissionResponse]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_datasets(
    session: SessionDep,
    auth_context: AuthContextDep,
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of records to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
) -> APIResponse[list[AssessmentSubmissionResponse]]:
    """List uploaded submission files."""
    submissions = list_submissions(
        session=session,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
        limit=limit,
        offset=offset,
    )

    return APIResponse.success_response(
        data=[_submission_to_response(submission) for submission in submissions]
    )


@router.get(
    "/datasets/{dataset_id}",
    description=load_description("assessment/get_dataset.md"),
    response_model=APIResponse[AssessmentSubmissionResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_dataset(
    dataset_id: UUID,
    session: SessionDep,
    auth_context: AuthContextDep,
    include_signed_url: bool = Query(
        False, description="Include a signed URL for downloading the raw file from S3"
    ),
    limit_rows: Annotated[
        int | None,
        Query(
            ge=1,
            le=100,
            description=(
                "If set, fetch the underlying file and include a preview of the "
                "first N data rows plus column headers. Skip to avoid the file "
                "download."
            ),
        ),
    ] = None,
) -> APIResponse[AssessmentSubmissionResponse]:
    """Get one uploaded submission file."""
    submission = get_submission_by_id(
        session=session,
        submission_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    signed_url = None
    if include_signed_url and submission.object_store_url:
        storage = get_cloud_storage(
            session=session, project_id=auth_context.project_.id
        )
        signed_url = storage.get_signed_url(submission.object_store_url)

    preview: AssessmentSubmissionPreview | None = None
    if limit_rows is not None:
        headers, rows = preview_submission(
            session=session,
            submission=submission,
            project_id=auth_context.project_.id,
            limit=limit_rows,
        )
        preview = AssessmentSubmissionPreview(
            headers=headers,
            rows=rows,
            returned_rows=len(rows),
            truncated=len(rows) >= limit_rows,
        )

    return APIResponse.success_response(
        data=_submission_to_response(submission, signed_url=signed_url, preview=preview)
    )


@router.delete(
    "/datasets/{dataset_id}",
    description=load_description("assessment/delete_dataset.md"),
    response_model=APIResponse[dict],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_dataset(
    dataset_id: UUID,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[dict]:
    """Delete an uploaded submission file."""
    submission = get_submission_by_id(
        session=session,
        submission_id=dataset_id,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    submission_name = submission.name
    error = delete_submission(session=session, submission=submission)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return APIResponse.success_response(
        data={
            "message": (
                f"Successfully deleted submission '{submission_name}' (id={dataset_id})"
            ),
            "submission_id": str(dataset_id),
        }
    )
