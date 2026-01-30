"""Main router for STT evaluation API routes."""

from fastapi import APIRouter

from . import dataset, evaluation, files, result

router = APIRouter(prefix="/evaluations/stt", tags=["STT Evaluation"])

# Include all sub-routers
router.include_router(files.router)
router.include_router(dataset.router)
router.include_router(evaluation.router)
router.include_router(result.router)
