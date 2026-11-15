from fastapi import APIRouter

from app.api.routes.documents.v2 import document

router = APIRouter()

router.include_router(document.router)
