from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import TokenDep, api_key_header
from app.api.permissions import Permission, require_permission
from app.core.security import ACCESS_TOKEN_COOKIE_NAME, API_KEY_HEADER_NAME
from app.models import AgentQueryRequest, AgentQueryResponse
from app.services.agent import (
    AgentLLMError,
    AgentNotConfiguredError,
    run_agent_query,
)
from app.services.agent.executor import AUTHORIZATION_HEADER
from app.utils import APIResponse, load_description

router = APIRouter(prefix="/agent", tags=["Agent"])


def _extract_forwardable_credentials(
    *, api_key: str | None, bearer_token: str | None, cookie_token: str | None
) -> dict[str, str]:
    credentials: dict[str, str] = {}
    if api_key:
        credentials[API_KEY_HEADER_NAME] = api_key

    token = bearer_token or cookie_token
    if token:
        credentials[AUTHORIZATION_HEADER] = f"Bearer {token}"
    return credentials


@router.post(
    "",
    description=load_description("agent/query.md"),
    response_model=APIResponse[AgentQueryResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def query_agent(
    request: Request,
    body: AgentQueryRequest,
    api_key: Annotated[str | None, Depends(api_key_header)],
    bearer_token: TokenDep,
) -> APIResponse[AgentQueryResponse]:
    try:
        result = await run_agent_query(
            query=body.query,
            forwarded_headers=_extract_forwardable_credentials(
                api_key=api_key,
                bearer_token=bearer_token,
                cookie_token=request.cookies.get(ACCESS_TOKEN_COOKIE_NAME),
            ),
            asgi_app=request.app,
        )
    except AgentNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentLLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return APIResponse.success_response(result)
