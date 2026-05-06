from fastapi import APIRouter

from app.api.routes.config import config, version

router = APIRouter(tags=["Config Management"])

router.include_router(config.router, prefix="/configs")
router.include_router(version.router, prefix="/configs")

__all__ = ["router"]
