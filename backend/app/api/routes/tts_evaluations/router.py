"""Main router for TTS evaluation API routes."""

from fastapi import APIRouter

from . import dataset, evaluation, result

router = APIRouter(prefix="/evaluations/tts", tags=["TTS Evaluation"])

# Include all sub-routers
router.include_router(dataset.router)
router.include_router(evaluation.router)
router.include_router(result.router)
