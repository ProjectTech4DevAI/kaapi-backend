import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import ChatbotMessage, ChatbotTurnRequest, ChatbotTurnResponse
from app.services.chatbot.engine import (
    ChatbotInputError,
    ChatbotProviderError,
    run_chatbot_turn,
)
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


# Synchronous by design (real-time request/response turn), not a Celery job.
@router.post(
    "/message",
    description=load_description("chatbot/message.md"),
    response_model=APIResponse[ChatbotTurnResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def send_chatbot_message(
    request: ChatbotTurnRequest,
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[ChatbotTurnResponse]:
    history: list[dict[str, str]] = []
    for entry in request.history:
        history.append(entry.model_dump(mode="json"))

    try:
        model_settings = resolve_chatbot_model_settings(
            session=session,
            project_id=auth_context.project_.id,
            config_id=request.config_id,
            config_version=request.config_version,
            max_tokens=request.max_tokens,
            history_token_limit=request.history_token_limit,
        )
        transcript = await run_chatbot_turn(
            system_prompt=request.system_prompt,
            history=history,
            user_message=request.message,
            model_settings=model_settings,
        )
    except ChatbotInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ChatbotProviderError as e:
        logger.warning(
            f"[send_chatbot_message] Chatbot turn failed upstream | "
            f"project_id: {auth_context.project_.id}"
        )
        raise HTTPException(status_code=502, detail=str(e))

    messages: list[ChatbotMessage] = []
    for entry in transcript:
        messages.append(ChatbotMessage.model_validate(entry))
    return APIResponse.success_response(ChatbotTurnResponse(messages=messages))
