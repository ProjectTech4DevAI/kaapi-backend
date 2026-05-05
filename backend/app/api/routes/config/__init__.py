from fastapi import APIRouter

from app.api.routes.config import config, version

router = APIRouter(prefix="/configs", tags=["Config Management"])

router.include_router(config.router)
router.include_router(version.router)

__all__ = ["router"]
