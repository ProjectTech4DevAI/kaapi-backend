import logging
import time

from fastapi import APIRouter, Depends

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.models import LLMCallRequest, LLMCallResponse, Message
from app.services.llm.jobs import start_job
from app.utils import APIResponse, validate_callback_url, load_description


logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])
llm_callback_router = APIRouter()


@llm_callback_router.post(
    "{$callback_url}",
    name="llm_callback",
)
def llm_callback_notification(body: APIResponse[LLMCallResponse]):
    """
    Callback endpoint specification for LLM call completion.

    The callback will receive:
    - On success: APIResponse with success=True and data containing LLMCallResponse
    - On failure: APIResponse with success=False and error message
    - metadata field will always be included if provided in the request
    """
    ...


@router.post(
    "/llm/call",
    description=load_description("llm/llm_call.md"),
    response_model=APIResponse[Message],
    callbacks=llm_callback_router.routes,
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
def llm_call(
    _current_user: AuthContextDep, session: SessionDep, request: LLMCallRequest
):
    """
    Endpoint to initiate an LLM call as a background job.
    """
    # ═══ START: End-to-End Timing ═══
    # Use time.time() for cross-process timing (wall-clock time)
    api_start_time_wall = time.time()
    # Use perf_counter for local API timing (higher precision)
    api_start_time_local = time.perf_counter()
    logger.info("[E2E_TIMING] ═══ API REQUEST RECEIVED ═══")

    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    t_validate_start = time.perf_counter()
    if request.callback_url:
        validate_callback_url(str(request.callback_url))
    t_validate = (time.perf_counter() - t_validate_start) * 1000

    t_job_start = time.perf_counter()
    job_id = start_job(
        db=session,
        request=request,
        project_id=project_id,
        organization_id=organization_id,
        api_start_time_wall=api_start_time_wall,  # Wall-clock time for cross-process timing
    )
    t_job = (time.perf_counter() - t_job_start) * 1000

    api_total_time = (time.perf_counter() - api_start_time_local) * 1000

    logger.info(
        f"[E2E_TIMING] API endpoint timing | "
        f"callback_validate={t_validate:.2f}ms, "
        f"start_job={t_job:.2f}ms, "
        f"total_api_time={api_total_time:.2f}ms | "
        f"job_id={job_id}"
    )

    return APIResponse.success_response(
        data=Message(
            message=f"Your response is being generated and will be delivered via callback."
        ),
    )
