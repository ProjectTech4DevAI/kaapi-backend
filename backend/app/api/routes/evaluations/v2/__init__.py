from fastapi import APIRouter

from app.api.routes.evaluations.v2 import (
    dataset,
    evaluation,
    iteration,
    prompt_improvement,
)

router = APIRouter()

router.include_router(evaluation.router)
router.include_router(dataset.router)
router.include_router(prompt_improvement.router)
router.include_router(iteration.router)
