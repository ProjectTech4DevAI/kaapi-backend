from fastapi import APIRouter, Depends

from app.api.permissions import require_feature
from app.api.routes.chatbot.message import router as message_router
from app.api.routes.chatbot.session import router as session_router
from app.core.feature_flags import FeatureFlag

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"],
    dependencies=[Depends(require_feature(FeatureFlag.CHATBOT))],
)
router.include_router(message_router)
router.include_router(session_router)
