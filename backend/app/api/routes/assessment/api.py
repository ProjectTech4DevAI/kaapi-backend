"""Assessment API-client route.

Run mode is inferred from the input shape:
    - RESPONSE: single object input (`input` key) — deferred, returns 501.
    - BATCH: submission-list input (`data` key) — N items over one provider batch series.

Results are delivered by webhook only (the request's required `callback_url`); there is no
status or result poll endpoint.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_feature, require_permission
from app.core.feature_flags import FeatureFlag
from app.models.assessment import (
    AssessmentCreate,
    AssessmentSubmitResponse,
    BatchInput,
)
from app.services.assessment.api import submission
from app.utils import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/assessments",
    tags=["Assessment (API)"],
    dependencies=[Depends(require_feature(FeatureFlag.ASSESSMENT))],
)

_RESPONSE_NOT_WIRED = "RESPONSE-mode assessment is not wired yet."


@router.post(
    "",
    response_model=APIResponse[AssessmentSubmitResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_assessment(
    request: AssessmentCreate,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[AssessmentSubmitResponse]:
    """Submit an assessment; the method is inferred from the input shape. The result is
    delivered to the request's `callback_url` on completion (webhook only)."""
    if not isinstance(request.input, BatchInput):
        raise HTTPException(status_code=501, detail=_RESPONSE_NOT_WIRED)

    result = submission.submit(
        session=session,
        request=request,
        organization_id=auth_context.organization_.id,
        project_id=auth_context.project_.id,
    )
    return APIResponse.success_response(data=result)
