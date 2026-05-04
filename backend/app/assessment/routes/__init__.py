from fastapi import APIRouter, Depends

from app.api.permissions import require_feature
from app.assessment.routes import assessments, datasets, runs
from app.core.feature_flags import FeatureFlag

router = APIRouter(
    prefix="/assessment",
    tags=["Assessment"],
    dependencies=[Depends(require_feature(FeatureFlag.ASSESSMENT))],
)

router.include_router(datasets.router)
router.include_router(assessments.router)
router.include_router(runs.router)

__all__ = ["router"]
