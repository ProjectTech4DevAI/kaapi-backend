from fastapi import APIRouter, Depends

from app.api.permissions import require_feature
from app.api.routes.assessment import datasets
from app.core.feature_flags import FeatureFlag

router = APIRouter(
    prefix="/assessment",
    tags=["Assessment"],
    dependencies=[Depends(require_feature(FeatureFlag.ASSESSMENT))],
)

# RUN routers (assessments, runs) are retired; their modules go in the follow-up PR.
router.include_router(datasets.router)

__all__ = ["router"]
