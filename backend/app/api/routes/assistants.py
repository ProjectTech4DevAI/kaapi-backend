from typing import Annotated

from fastapi import APIRouter, Depends, Path, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.crud import (
    fetch_assistant_from_openai,
    sync_assistant,
    create_assistant,
    update_assistant,
    get_assistant_by_id,
    get_assistants_by_project,
    delete_assistant,
)
from app.models import AssistantCreate, AssistantUpdate, Assistant
from app.api.permissions import Permission, require_permission
from app.utils import APIResponse, get_openai_client

router = APIRouter(prefix="/assistant", tags=["Assistants"])


@router.post(
    "/{assistant_id}/ingest",
    response_model=APIResponse[Assistant],
    status_code=201,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def ingest_assistant_route(
    assistant_id: Annotated[str, Path(description="The ID of the assistant to ingest")],
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Ingest an assistant from OpenAI and store it in the platform.
    """

    client = get_openai_client(
        session, current_user.organization_.id, current_user.project_.id
    )

    openai_assistant = fetch_assistant_from_openai(assistant_id, client)
    assistant = sync_assistant(
        session=session,
        organization_id=current_user.organization_.id,
        project_id=current_user.project_.id,
        openai_assistant=openai_assistant,
    )

    return APIResponse.success_response(assistant)


@router.post(
    "/",
    response_model=APIResponse[Assistant],
    status_code=201,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def create_assistant_route(
    assistant_in: AssistantCreate,
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Create a new assistant in the local DB, checking that vector store IDs exist in OpenAI first.
    """
    client = get_openai_client(
        session, current_user.organization_.id, current_user.project_.id
    )
    assistant = create_assistant(
        session=session,
        openai_client=client,
        assistant=assistant_in,
        project_id=current_user.project_.id,
        organization_id=current_user.organization_.id,
    )
    return APIResponse.success_response(assistant)


@router.patch(
    "/{assistant_id}",
    response_model=APIResponse[Assistant],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def update_assistant_route(
    assistant_id: Annotated[str, Path(description="Assistant ID to update")],
    assistant_update: AssistantUpdate,
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Update an existing assistant with provided fields. Supports replacing, adding, or removing vector store IDs.
    """
    client = get_openai_client(
        session, current_user.organization_.id, current_user.project_.id
    )
    updated_assistant = update_assistant(
        session=session,
        assistant_id=assistant_id,
        openai_client=client,
        project_id=current_user.project_.id,
        assistant_update=assistant_update,
    )
    return APIResponse.success_response(updated_assistant)


@router.get(
    "/{assistant_id}",
    response_model=APIResponse[Assistant],
    summary="Get a single assistant by its ID",
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def get_assistant_route(
    assistant_id: Annotated[str, Path(description="The assistant_id to fetch")],
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Fetch a single assistant by its assistant_id.
    """
    assistant = get_assistant_by_id(session, assistant_id, current_user.project_.id)
    if not assistant:
        raise HTTPException(
            status_code=404, detail=f"Assistant with ID {assistant_id} not found."
        )
    return APIResponse.success_response(assistant)


@router.get(
    "/",
    response_model=APIResponse[list[Assistant]],
    summary="List all assistants in the current project",
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def list_assistants_route(
    session: SessionDep,
    current_user: AuthContextDep,
    skip: int = Query(0, ge=0, description="How many items to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum items to return"),
):
    """
    List all assistants in the current project and organization.
    """

    assistants = get_assistants_by_project(
        session=session, project_id=current_user.project_.id, skip=skip, limit=limit
    )
    return APIResponse.success_response(assistants)


@router.delete(
    "/{assistant_id}",
    response_model=APIResponse,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def delete_assistant_route(
    assistant_id: Annotated[str, Path(description="Assistant ID to delete")],
    session: SessionDep,
    current_user: AuthContextDep,
):
    """
    Soft delete an assistant by marking it as deleted.
    """
    delete_assistant(
        session=session,
        assistant_id=assistant_id,
        project_id=current_user.project_.id,
    )
    return APIResponse.success_response(
        data={"message": "Assistant deleted successfully."}
    )
