"""Evaluation API routes."""

from fastapi import APIRouter

from app.api.routes.evaluations import dataset, evaluation

router = APIRouter(prefix="/evaluations", tags=["evaluation"])

# Include dataset routes under /evaluations/datasets
router.include_router(dataset.router, prefix="/datasets")

# Include evaluation routes directly under /evaluations
router.include_router(evaluation.router)
