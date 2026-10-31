from fastapi import APIRouter, Depends

from app.api.permissions import require_feature
from app.api.routes.chatbot.chatbot import router as chatbot_router
from app.api.routes.chatbot.flow import router as flow_router
from app.api.routes.chatbot.session import router as session_router
from app.core.feature_flags import FeatureFlag

CHATBOTS_PREFIX = "/chatbots"

router = APIRouter(
    tags=["Chatbot"],
    dependencies=[Depends(require_feature(FeatureFlag.CHATBOT))],
)
router.include_router(session_router, prefix=CHATBOTS_PREFIX)
router.include_router(chatbot_router, prefix=CHATBOTS_PREFIX)
router.include_router(flow_router, prefix=CHATBOTS_PREFIX)
