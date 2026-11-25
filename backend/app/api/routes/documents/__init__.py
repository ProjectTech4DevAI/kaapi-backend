from fastapi import APIRouter

from app.api.routes.documents import document

router = APIRouter()

router.include_router(document.router)
