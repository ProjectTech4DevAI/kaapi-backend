"""Main router for evaluation API routes."""

from fastapi import APIRouter

from app.api.routes.evaluations import dataset, evaluation
from app.api.routes.stt_evaluations.router import router as stt_router
from app.api.routes.tts_evaluations.router import router as tts_router

router = APIRouter()

router.include_router(dataset.router)
router.include_router(stt_router)
router.include_router(tts_router)
router.include_router(evaluation.router)
