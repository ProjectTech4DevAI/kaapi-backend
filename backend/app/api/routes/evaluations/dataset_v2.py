"""v2 Langfuse-free evaluation dataset upload route.

Replica of the v1 dataset upload with the same multipart shape and response, but
it creates no Langfuse dataset and stores only the original items — duplication is
recorded in metadata and applied at run time.
"""

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.api.routes.evaluations.dataset import _dataset_to_response
from app.core.rate_monitor import monitor_rate
from app.models.evaluation import DatasetUploadResponse
from app.services.evaluations import upload_dataset_v2, validate_csv_file
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations/datasets", tags=["Evaluation v2"])


@router.post(
    "",
    description=load_description("evaluation/create_evaluation_dataset_v2.md"),
    response_model=APIResponse[DatasetUploadResponse],
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("evaluations")),
    ],
)
async def upload_dataset_v2_route(
    session: SessionDep,
    auth_context: AuthContextDep,
    file: UploadFile = File(
        ..., description="CSV file with 'question' and 'answer' columns"
    ),
    dataset_name: str = Form(..., description="Name for the dataset"),
    description: str | None = Form(None, description="Optional dataset description"),
    duplication_factor: int = Form(
        default=1,
        ge=1,
        le=5,
        description="Run-time duplication per item (min: 1, max: 5)",
    ),
) -> APIResponse[DatasetUploadResponse]:
    """Upload a Langfuse-free evaluation dataset (v2)."""
    csv_content = await validate_csv_file(file)

    dataset = upload_dataset_v2(
        session=session,
        csv_content=csv_content,
        dataset_name=dataset_name,
        description=description,
        duplication_factor=duplication_factor,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )

    return APIResponse.success_response(data=_dataset_to_response(dataset))
