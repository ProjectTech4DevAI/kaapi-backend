from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import ChatbotMessage, ChatbotTurnRequest, ChatbotTurnResponse
from app.services.chatbot.agents.onboarding.graph import run_chatbot_turn
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.utils import APIResponse, load_description

router = APIRouter()


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
        turn_result = await run_chatbot_turn(
            system_prompt=request.system_prompt,
            history=history,
            user_message=request.message,
            model_settings=model_settings,
        )
    except ChatbotInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ChatbotProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    messages: list[ChatbotMessage] = []
    for entry in turn_result.transcript:
        messages.append(ChatbotMessage.model_validate(entry))
    return APIResponse.success_response(
        ChatbotTurnResponse(
            messages=messages,
            collected_fields=turn_result.collected_fields,
            completion_status=turn_result.completion_status,
        )
    )
