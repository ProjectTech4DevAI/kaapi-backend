from fastapi import APIRouter

from app.assessment.routes import assessments, datasets, runs

router = APIRouter(prefix="/assessment", tags=["Assessment"])

router.include_router(datasets.router)
router.include_router(assessments.router)
router.include_router(runs.router)

__all__ = ["router"]
