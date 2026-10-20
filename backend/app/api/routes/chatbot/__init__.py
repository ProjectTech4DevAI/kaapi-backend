from fastapi import APIRouter

from app.api.routes.chatbot.message import router as message_router

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])
router.include_router(message_router)
