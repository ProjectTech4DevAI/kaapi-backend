import re
import openai
import requests

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.core import logging, settings
from app.models import OpenAIThreadCreate
from app.crud import upsert_thread_result, get_thread_result
from app.utils import APIResponse, mask_string
from app.crud.credentials import get_provider_credential
from app.core.util import configure_openai
from app.core.langfuse.langfuse import LangfuseTracer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Threads"])


class StartThreadRequest(BaseModel):
    question: str = Field(..., description="The user's input question.")
    assistant_id: str = Field(..., description="The ID of the assistant to be used.")
    remove_citation: bool = Field(
        default=False, description="Whether to remove citations from the response."
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="An optional existing thread ID to continue the conversation.",
    )


def send_callback(callback_url: str, data: dict):
    """Send results to the callback URL (synchronously)."""
    try:
        session = requests.Session()
        # uncomment this to run locally without SSL
        # session.verify = False
        response = session.post(callback_url, json=data)
        response.raise_for_status()
        logger.info(f"[send_callback] Callback sent successfully to {callback_url}")
        return True
    except requests.RequestException as e:
        logger.error(f"[send_callback] Callback failed: {str(e)}", exc_info=True)
        return False


def handle_openai_error(e: openai.OpenAIError) -> str:
    """Extract error message from OpenAI error."""
    if isinstance(e.body, dict) and "message" in e.body:
        return e.body["message"]
    return str(e)


def validate_thread(client: OpenAI, thread_id: str) -> tuple[bool, str]:
    """Validate if a thread exists and has no active runs."""
    if not thread_id:
        return True, None

    try:
        runs = client.beta.threads.runs.list(thread_id=thread_id)
        if runs.data and len(runs.data) > 0:
            latest_run = runs.data[0]
            if latest_run.status in ["queued", "in_progress", "requires_action"]:
                logger.error(
                    f"[validate_thread] Thread ID {mask_string(thread_id)} is currently {latest_run.status}."
                )
                return (
                    False,
                    f"There is an active run on this thread (status: {latest_run.status}). Please wait for it to complete.",
                )
        return True, None
    except openai.OpenAIError as e:
        logger.error(
            f"[validate_thread] Failed to validate thread ID {mask_string(thread_id)}: {str(e)}",
            exc_info=True,
        )
        return False, f"Invalid thread ID provided {thread_id}"


def setup_thread(client: OpenAI, request: dict) -> tuple[bool, str]:
    """Set up thread and add message, either creating new or using existing."""
    thread_id = request.get("thread_id")
    if thread_id:
        try:
            client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=request["question"]
            )
            logger.info(
                f"[setup_thread] Added message to existing thread {mask_string(thread_id)}"
            )
            return True, None
        except openai.OpenAIError as e:
            logger.error(
                f"[setup_thread] Failed to add message to existing thread {mask_string(thread_id)}: {str(e)}",
                exc_info=True,
            )
            return False, handle_openai_error(e)
    else:
        try:
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=request["question"]
            )
            request["thread_id"] = thread.id
            logger.info(
                f"[setup_thread] Created new thread with ID: {mask_string(thread.id)}"
            )
            return True, None
        except openai.OpenAIError as e:
            logger.error(
                f"[setup_thread] Failed to create new thread: {str(e)}", exc_info=True
            )
            return False, handle_openai_error(e)


def process_message_content(message_content: str, remove_citation: bool) -> str:
    """Process message content, optionally removing citations."""
    if remove_citation:
        return re.sub(r"【\d+(?::\d+)?†[^】]*】", "", message_content)
    return message_content


def get_additional_data(request: dict) -> dict:
    """Extract additional data from request, excluding specific keys."""
    return {
        k: v
        for k, v in request.items()
        if k not in {"question", "assistant_id", "callback_url", "thread_id"}
    }


def create_success_response(request: dict, message: str) -> APIResponse:
    """Create a success response with the given message and request data."""
    additional_data = get_additional_data(request)
    return APIResponse.success_response(
        data={
            "status": "success",
            "message": message,
            "thread_id": request["thread_id"],
            "endpoint": getattr(request, "endpoint", "some-default-endpoint"),
            **additional_data,
        }
    )


def run_and_poll_thread(client: OpenAI, thread_id: str, assistant_id: str):
    """Runs and polls a thread with the specified assistant using the OpenAI client."""
    return client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


def extract_response_from_thread(
    client: OpenAI, thread_id: str, remove_citation: bool = False
) -> str:
    """Fetches and processes the latest message from a thread."""
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    latest_message = messages.data[0]
    message_content = latest_message.content[0].text.value
    return process_message_content(message_content, remove_citation)


def process_run_core(
    request: dict, client: OpenAI, tracer: LangfuseTracer
) -> tuple[dict, str]:
    """Core function to process a run and return the response and message with Langfuse tracing."""
    tracer.start_generation(
        name="openai_thread_run",
        input={"question": request["question"]},
        metadata={"assistant_id": request["assistant_id"]},
    )

    try:
        logger.info(
            f"[process_run_core] Starting run for thread ID: {mask_string(request.get('thread_id'))} with assistant ID: {mask_string(request.get('assistant_id'))}"
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=request["thread_id"],
            assistant_id=request["assistant_id"],
        )

        if run.status == "completed":
            message = extract_response_from_thread(
                client, request["thread_id"], request.get("remove_citation", False)
            )
            tracer.end_generation(
                output={
                    "thread_id": request["thread_id"],
                    "message": message,
                },
                usage={
                    "input": run.usage.prompt_tokens,
                    "output": run.usage.completion_tokens,
                    "total": run.usage.total_tokens,
                    "unit": "TOKENS",
                },
                model=run.model,
            )
            tracer.update_trace(
                tags=[request["thread_id"]],
                output={"status": "success", "message": message, "error": None},
            )
            diagnostics = {
                "input_tokens": run.usage.prompt_tokens,
                "output_tokens": run.usage.completion_tokens,
                "total_tokens": run.usage.total_tokens,
                "model": run.model,
            }
            request = {**request, **{"diagnostics": diagnostics}}
            logger.info(
                f"[process_run_core] Run completed successfully for thread ID: {mask_string(request.get('thread_id'))}"
            )
            return create_success_response(request, message).model_dump(), None
        else:
            error_msg = f"Run failed with status: {run.status}"
            logger.error(
                f"[process_run_core] Run failed with error: {run.last_error} for thread ID: {mask_string(request.get('thread_id'))}"
            )
            tracer.log_error(error_msg)
            return APIResponse.failure_response(error=error_msg).model_dump(), error_msg

    except openai.OpenAIError as e:
        error_msg = handle_openai_error(e)
        tracer.log_error(error_msg)
        logger.error(
            f"[process_run_core] OpenAI error: {error_msg} for thread ID: {mask_string(request.get('thread_id'))}",
            exc_info=True,
        )
        return APIResponse.failure_response(error=error_msg).model_dump(), error_msg
    finally:
        tracer.flush()


def process_run(request: dict, client: OpenAI, tracer: LangfuseTracer):
    """Process a run and send callback with results with Langfuse tracing."""
    response, _ = process_run_core(request, client, tracer)
    send_callback(request["callback_url"], response)


def poll_run_and_prepare_response(request: dict, client: OpenAI, db: SessionDep):
    """Handles a thread run, processes the response, and upserts the result to the database."""
    thread_id = request["thread_id"]
    prompt = request["question"]

    logger.info(
        f"[poll_run_and_prepare_response] Starting run for thread ID: {mask_string(thread_id)}"
    )

    try:
        run = run_and_poll_thread(client, thread_id, request["assistant_id"])
        status = run.status or "unknown"
        response = None
        error = None

        if status == "completed":
            response = extract_response_from_thread(
                client, thread_id, request.get("remove_citation", False)
            )
            logger.info(
                f"[poll_run_and_prepare_response] Successfully executed run for thread ID: {mask_string(thread_id)}"
            )

    except openai.OpenAIError as e:
        status = "failed"
        error = str(e)
        response = None
        logger.error(
            f"[poll_run_and_prepare_response] Run failed for thread ID {mask_string(thread_id)}: {error}",
            exc_info=True,
        )

    upsert_thread_result(
        db,
        OpenAIThreadCreate(
            thread_id=thread_id,
            prompt=prompt,
            response=response,
            status=status,
            error=error,
        ),
    )


@router.post(
    "/threads",
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
async def threads(
    request: dict,
    background_tasks: BackgroundTasks,
    _session: SessionDep,
    _current_user: AuthContextDep,
):
    """Asynchronous endpoint that processes requests in background."""
    credentials = get_provider_credential(
        session=_session,
        org_id=_current_user.organization_.id,
        provider="openai",
        project_id=request.get("project_id"),
    )
    client, success = configure_openai(credentials)
    if not success:
        logger.error(
            f"[threads] OpenAI API key not configured for this organization. | organization_id: {_current_user.organization_.id}, project_id: {request.get('project_id')}"
        )
        return APIResponse.failure_response(
            error="OpenAI API key not configured for this organization."
        )

    langfuse_credentials = get_provider_credential(
        session=_session,
        org_id=_current_user.organization_.id,
        provider="langfuse",
        project_id=request.get("project_id"),
    )

    # Validate thread
    is_valid, error_message = validate_thread(client, request.get("thread_id"))
    if not is_valid:
        raise Exception(error_message)
    # Setup thread
    is_success, error_message = setup_thread(client, request)
    if not is_success:
        raise Exception(error_message)

    # Send immediate response
    initial_response = APIResponse.success_response(
        data={
            "status": "processing",
            "message": "Run started",
            "thread_id": request.get("thread_id"),
            "success": True,
        }
    )

    tracer = LangfuseTracer(
        credentials=langfuse_credentials,
        session_id=request.get("thread_id"),
    )

    tracer.start_trace(
        name="threads_async_endpoint",
        input={
            "question": request["question"],
            "assistant_id": request["assistant_id"],
        },
        metadata={"thread_id": request["thread_id"]},
    )
    # Schedule background task
    background_tasks.add_task(process_run, request, client, tracer)
    logger.info(
        f"[threads] Background task scheduled for thread ID: {mask_string(request.get('thread_id'))} | organization_id: {_current_user.organization_.id}, project_id: {request.get('project_id')}"
    )
    return initial_response


@router.post(
    "/threads/sync",
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
async def threads_sync(
    request: dict,
    _session: SessionDep,
    _current_user: AuthContextDep,
):
    """Synchronous endpoint that processes requests immediately."""
    credentials = get_provider_credential(
        session=_session,
        org_id=_current_user.organization_.id,
        provider="openai",
        project_id=request.get("project_id"),
    )

    # Configure OpenAI client
    client, success = configure_openai(credentials)
    if not success:
        logger.error(
            f"[threads_sync] OpenAI API key not configured for this organization. | organization_id: {_current_user.organization_.id}, project_id: {request.get('project_id')}"
        )
        return APIResponse.failure_response(
            error="OpenAI API key not configured for this organization."
        )

    # Get Langfuse credentials
    langfuse_credentials = get_provider_credential(
        session=_session,
        org_id=_current_user.organization_.id,
        provider="langfuse",
        project_id=request.get("project_id"),
    )

    # Validate thread
    is_valid, error_message = validate_thread(client, request.get("thread_id"))
    if not is_valid:
        raise Exception(error_message)

    # Setup thread
    is_success, error_message = setup_thread(client, request)
    if not is_success:
        raise Exception(error_message)

    tracer = LangfuseTracer(
        credentials=langfuse_credentials,
        session_id=request.get("thread_id"),
    )

    tracer.start_trace(
        name="threads_sync_endpoint",
        input={
            "question": request.get("question"),
            "assistant_id": request.get("assistant_id"),
        },
        metadata={"thread_id": request.get("thread_id")},
    )

    response, error_message = process_run_core(request, client, tracer)
    return response


@router.post(
    "/threads/start",
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
)
async def start_thread(
    request: StartThreadRequest,
    background_tasks: BackgroundTasks,
    db: SessionDep,
    _current_user: AuthContextDep,
):
    """
    Create a new OpenAI thread for the given question and start polling in the background.
    """
    request = request.model_dump()
    prompt = request["question"]
    credentials = get_provider_credential(
        session=db,
        org_id=_current_user.organization_.id,
        provider="openai",
        project_id=_current_user.project_.id,
    )

    # Configure OpenAI client
    client, success = configure_openai(credentials)
    if not success:
        logger.error(
            f"[start_thread] OpenAI API key not configured for this organization. | project_id: {_current_user.project_.id}"
        )
        return APIResponse.failure_response(
            error="OpenAI API key not configured for this organization."
        )

    is_success, error = setup_thread(client, request)
    if not is_success:
        raise Exception(error)

    thread_id = request["thread_id"]

    upsert_thread_result(
        db,
        OpenAIThreadCreate(
            thread_id=thread_id,
            prompt=prompt,
            response=None,
            status="processing",
            error=None,
        ),
    )

    background_tasks.add_task(poll_run_and_prepare_response, request, client, db)

    logger.info(
        f"[start_thread] Background task scheduled to process response for thread ID: {mask_string(thread_id)} | project_id: {_current_user.project_.id}"
    )
    return APIResponse.success_response(
        data={
            "thread_id": thread_id,
            "prompt": prompt,
            "status": "processing",
            "message": "Thread created and polling started in background.",
        }
    )


@router.get(
    "/threads/result/{thread_id}",
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
async def get_thread(
    thread_id: str,
    db: SessionDep,
    _current_user: AuthContextDep,
):
    """
    Retrieve the result of a previously started OpenAI thread using its thread ID.
    """
    result = get_thread_result(db, thread_id)

    if not result:
        logger.error(
            f"[get_thread] Thread result not found for ID: {mask_string(thread_id)} | org_id: {_current_user.organization_.id}"
        )
        raise HTTPException(404, "thread not found")

    status = result.status or ("success" if result.response else "processing")

    return APIResponse.success_response(
        data={
            "thread_id": result.thread_id,
            "prompt": result.prompt,
            "status": status,
            "response": result.response,
            "error": result.error,
        }
    )
