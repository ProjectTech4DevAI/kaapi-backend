import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace

from app.api.deps import AuthContextDep, SessionDep
from app.api.docs.guardrails_schemas import (
    BAN_LIST_CREATE_EXAMPLE,
    BAN_LIST_LIST_OK,
    BAN_LIST_OK,
    BAN_LIST_UPDATE_EXAMPLE,
    DELETE_OK,
    JOB_ITEM_RESPONSES,
    JOB_RESPONSES,
    LLM_PROMPT_CONFIG_CREATE_EXAMPLES,
    LLM_PROMPT_CONFIG_LIST_OK,
    LLM_PROMPT_CONFIG_OK,
    LLM_PROMPT_CONFIG_UPDATE_EXAMPLE,
    PROXY_ITEM_RESPONSES,
    PROXY_ITEM_WRITE_RESPONSES,
    PROXY_READ_RESPONSES,
    PROXY_WRITE_RESPONSES,
    VALIDATOR_CONFIG_CREATE_EXAMPLES,
    VALIDATOR_CONFIG_LIST_OK,
    VALIDATOR_CONFIG_OK,
    VALIDATOR_CONFIG_UPDATE_EXAMPLE,
    VALIDATOR_TYPE_LIST_OK,
    render_validator_reference,
)
from app.api.openapi_docs import doc, ref
from app.api.permissions import Permission, require_permission
from app.core.rate_monitor import monitor_rate
from app.core.telemetry import log_context
from app.crud.jobs import JobCrud
from app.models import JobStatus, JobType
from app.models.guardrails import (
    GuardrailsCallbackData,
    GuardrailsJobImmediatePublic,
    GuardrailsJobPublic,
    GuardrailsRequest,
)
from app.services.guardrails.jobs import start_job
from app.services.llm.guardrails import proxy_guardrails_request
from app.utils import APIResponse, load_description, validate_callback_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Guardrails"])
guardrails_callback_router = APIRouter()

ProxyBody = Annotated[
    Any,
    Body(
        description=(
            "Forwarded to the guardrails service unchanged. That service owns "
            "validation for this payload, so the gateway neither checks nor "
            "reshapes it; its 422 is what you see on a bad body."
        )
    ),
]
"""Body of a management-API proxy route.

Deliberately untyped at run time: a Pydantic mirror of the upstream schema
would reject payloads upstream accepts the moment the two drift (a new
validator type, a new tuning key). The documented shape lives in
`app.api.docs.guardrails_schemas` instead.
"""


@guardrails_callback_router.post(
    "{$callback_url}",
    name="guardrails_callback",
)
def guardrails_callback_notification(
    body: APIResponse[GuardrailsCallbackData],
) -> None:
    """Callback delivered to `callback_url` when a guardrails job finishes.

    On success `success=True` and `data` carries the sanitised text. On a hard
    block `success=False`, `error` carries the upstream message and `data` is
    null. `metadata` echoes the request's `request_metadata` plus a
    server-managed `warnings` list.
    """


@router.post(
    "/guardrails",
    summary="Apply guardrails to text",
    description=load_description("guardrails/apply_guardrails.md"),
    response_model=APIResponse[GuardrailsJobImmediatePublic],
    callbacks=guardrails_callback_router.routes,
    dependencies=[
        Depends(require_permission(Permission.REQUIRE_PROJECT)),
        Depends(monitor_rate("llm_call")),
    ],
    **doc(responses=JOB_RESPONSES),
)
def apply_guardrails_endpoint(
    _current_user: AuthContextDep,
    session: SessionDep,
    request: GuardrailsRequest,
) -> APIResponse[GuardrailsJobImmediatePublic]:
    """Initiate a guardrails-only job. Returns the job_id immediately; the
    sanitised text is delivered via callback_url (or polled via GET)."""
    project_id = _current_user.project_.id
    organization_id = _current_user.organization_.id

    with log_context(
        tag="guardrails",
        system="guardrails",
        lifecycle="api.guardrails.apply",
        project_id=project_id,
        organization_id=organization_id,
        callback_enabled=request.callback_url is not None,
    ):
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("kaapi.project_id", project_id)
            span.set_attribute("kaapi.organization_id", organization_id)
            span.set_attribute(
                "guardrails.callback_enabled", request.callback_url is not None
            )

        if request.callback_url:
            validate_callback_url(str(request.callback_url))

        job = start_job(
            db=session,
            request=request,
            project_id=project_id,
            organization_id=organization_id,
        )

        if span.is_recording():
            span.set_attribute("guardrails.job_id", str(job.id))

        message = (
            "Guardrails are being applied; the sanitised text will be delivered via callback."
            if request.callback_url
            else "Guardrails are being applied; poll GET /guardrails/{job_id} for the result."
        )

        return APIResponse.success_response(
            data=GuardrailsJobImmediatePublic(
                job_id=job.id,
                status=job.status.value,
                message=message,
                job_inserted_at=job.inserted_at,
                job_updated_at=job.updated_at,
            )
        )


def _upstream_response(status_code: int, payload: Any) -> Response:
    """An empty upstream body must stay empty (204s cannot carry one).

    Tenant-scoped data must never be cached by a shared/intermediary cache
    (CWE-525); `no-store` is stronger than `private` for that guarantee.
    """
    headers = {"Cache-Control": "no-store"}
    if payload is None:
        return Response(status_code=status_code, headers=headers)
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


# ROUTE ORDERING: these fixed paths must stay above GET /guardrails/{job_id}.


@router.get(
    "/guardrails",
    summary="List validator types",
    # The per-type reference is rendered from the same catalogue that builds
    # the example response, so the two cannot disagree.
    description=load_description("guardrails/list_validator_types.md")
    + render_validator_reference(),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: VALIDATOR_TYPE_LIST_OK, **PROXY_READ_RESPONSES}),
)
def list_guardrails_validator_types(_current_user: AuthContextDep) -> Response:
    """List the validator types supported upstream and their JSON schemas."""
    status_code, payload = proxy_guardrails_request(
        "GET",
        "/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.post(
    "/guardrails/ban_lists",
    summary="Create a ban list",
    description=load_description("guardrails/create_ban_list.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(
        request_schema=ref("GuardrailsBanListCreate"),
        request=BAN_LIST_CREATE_EXAMPLE,
        responses={200: BAN_LIST_OK, **PROXY_WRITE_RESPONSES},
    ),
)
def create_guardrails_ban_list(
    _current_user: AuthContextDep, body: ProxyBody = None
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "POST",
        "/ban_lists/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        json_body=body,
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/ban_lists",
    summary="List ban lists",
    description=load_description("guardrails/list_ban_lists.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: BAN_LIST_LIST_OK, **PROXY_READ_RESPONSES}),
)
def list_guardrails_ban_lists(
    _current_user: AuthContextDep,
    domain: Annotated[
        str | None,
        Query(description="Filter to ban lists carrying this domain label."),
    ] = None,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
    limit: Annotated[
        int | None,
        Query(ge=1, le=100, description="Max rows to return. Unset means no limit."),
    ] = None,
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "GET",
        "/ban_lists/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        params={"domain": domain, "offset": offset, "limit": limit},
    )
    return _upstream_response(status_code, payload)


@router.post(
    "/guardrails/llm_prompt_configs",
    summary="Create an LLM prompt config",
    description=load_description("guardrails/create_llm_prompt_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(
        request_schema=ref("GuardrailsLLMPromptConfigCreate"),
        request_examples=LLM_PROMPT_CONFIG_CREATE_EXAMPLES,
        responses={200: LLM_PROMPT_CONFIG_OK, **PROXY_WRITE_RESPONSES},
    ),
)
def create_guardrails_llm_prompt_config(
    _current_user: AuthContextDep, body: ProxyBody = None
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "POST",
        "/llm_prompt_configs/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        json_body=body,
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/llm_prompt_configs",
    summary="List LLM prompt configs",
    description=load_description("guardrails/list_llm_prompt_configs.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: LLM_PROMPT_CONFIG_LIST_OK, **PROXY_READ_RESPONSES}),
)
def list_guardrails_llm_prompt_configs(
    _current_user: AuthContextDep,
    validator_name: Annotated[
        str | None,
        Query(
            description=(
                "Filter to prompts driving this validator, e.g. "
                "`topic_relevance` or `answer_relevance_custom_llm`."
            )
        ),
    ] = None,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
    limit: Annotated[
        int | None,
        Query(ge=1, le=100, description="Max rows to return. Unset means no limit."),
    ] = None,
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "GET",
        "/llm_prompt_configs/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        params={
            "validator_name": validator_name,
            "offset": offset,
            "limit": limit,
        },
    )
    return _upstream_response(status_code, payload)


@router.post(
    "/guardrails/validators/configs",
    summary="Create a validator config",
    description=load_description("guardrails/create_validator_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(
        request_schema=ref("GuardrailsValidatorConfigCreate"),
        request_examples=VALIDATOR_CONFIG_CREATE_EXAMPLES,
        responses={200: VALIDATOR_CONFIG_OK, **PROXY_WRITE_RESPONSES},
    ),
)
def create_guardrails_validator_config(
    _current_user: AuthContextDep, body: ProxyBody = None
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "POST",
        "/validators/configs/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        json_body=body,
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/validators/configs",
    summary="List validator configs",
    description=load_description("guardrails/list_validator_configs.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: VALIDATOR_CONFIG_LIST_OK, **PROXY_READ_RESPONSES}),
)
def list_guardrails_validator_configs(
    _current_user: AuthContextDep,
    ids: Annotated[
        list[UUID] | None,
        Query(description="Repeat to fetch several configs by id."),
    ] = None,
    stage: Annotated[
        str | None,
        Query(description="Filter by the stage the config targets: input or output."),
    ] = None,
    type: Annotated[
        str | None,
        Query(
            description=(
                "Filter by validator type. See GET /api/v1/guardrails for the "
                "supported types."
            )
        ),
    ] = None,
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "GET",
        "/validators/configs/",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        params={
            "ids": [str(config_id) for config_id in ids] if ids else None,
            "stage": stage,
            "type": type,
        },
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/validators/configs/{config_id}",
    summary="Get a validator config",
    description=load_description("guardrails/get_validator_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: VALIDATOR_CONFIG_OK, **PROXY_ITEM_RESPONSES}),
)
def get_guardrails_validator_config(
    _current_user: AuthContextDep,
    config_id: Annotated[UUID, Path(description="Validator config id.")],
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "GET",
        f"/validators/configs/{config_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.patch(
    "/guardrails/validators/configs/{config_id}",
    summary="Update a validator config",
    description=load_description("guardrails/update_validator_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(
        request_schema=ref("GuardrailsValidatorConfigUpdate"),
        request=VALIDATOR_CONFIG_UPDATE_EXAMPLE,
        responses={200: VALIDATOR_CONFIG_OK, **PROXY_ITEM_WRITE_RESPONSES},
    ),
)
def update_guardrails_validator_config(
    _current_user: AuthContextDep,
    config_id: Annotated[UUID, Path(description="Validator config id.")],
    body: ProxyBody = None,
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "PATCH",
        f"/validators/configs/{config_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        json_body=body,
    )
    return _upstream_response(status_code, payload)


@router.delete(
    "/guardrails/validators/configs/{config_id}",
    summary="Delete a validator config",
    description=load_description("guardrails/delete_validator_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: DELETE_OK, **PROXY_ITEM_RESPONSES}),
)
def delete_guardrails_validator_config(
    _current_user: AuthContextDep,
    config_id: Annotated[UUID, Path(description="Validator config id.")],
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "DELETE",
        f"/validators/configs/{config_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/ban_lists/{ban_list_id}",
    summary="Get a ban list",
    description=load_description("guardrails/get_ban_list.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: BAN_LIST_OK, **PROXY_ITEM_RESPONSES}),
)
def get_guardrails_ban_list(
    _current_user: AuthContextDep,
    ban_list_id: Annotated[UUID, Path(description="Ban list id.")],
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "GET",
        f"/ban_lists/{ban_list_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.patch(
    "/guardrails/ban_lists/{ban_list_id}",
    summary="Update a ban list",
    description=load_description("guardrails/update_ban_list.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(
        request_schema=ref("GuardrailsBanListUpdate"),
        request=BAN_LIST_UPDATE_EXAMPLE,
        responses={200: BAN_LIST_OK, **PROXY_ITEM_WRITE_RESPONSES},
    ),
)
def update_guardrails_ban_list(
    _current_user: AuthContextDep,
    ban_list_id: Annotated[UUID, Path(description="Ban list id.")],
    body: ProxyBody = None,
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "PATCH",
        f"/ban_lists/{ban_list_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        json_body=body,
    )
    return _upstream_response(status_code, payload)


@router.delete(
    "/guardrails/ban_lists/{ban_list_id}",
    summary="Delete a ban list",
    description=load_description("guardrails/delete_ban_list.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: DELETE_OK, **PROXY_ITEM_RESPONSES}),
)
def delete_guardrails_ban_list(
    _current_user: AuthContextDep,
    ban_list_id: Annotated[UUID, Path(description="Ban list id.")],
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "DELETE",
        f"/ban_lists/{ban_list_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/llm_prompt_configs/{prompt_config_id}",
    summary="Get an LLM prompt config",
    description=load_description("guardrails/get_llm_prompt_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: LLM_PROMPT_CONFIG_OK, **PROXY_ITEM_RESPONSES}),
)
def get_guardrails_llm_prompt_config(
    _current_user: AuthContextDep,
    prompt_config_id: Annotated[UUID, Path(description="LLM prompt config id.")],
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "GET",
        f"/llm_prompt_configs/{prompt_config_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.patch(
    "/guardrails/llm_prompt_configs/{prompt_config_id}",
    summary="Update an LLM prompt config",
    description=load_description("guardrails/update_llm_prompt_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(
        request_schema=ref("GuardrailsLLMPromptConfigUpdate"),
        request=LLM_PROMPT_CONFIG_UPDATE_EXAMPLE,
        responses={200: LLM_PROMPT_CONFIG_OK, **PROXY_ITEM_WRITE_RESPONSES},
    ),
)
def update_guardrails_llm_prompt_config(
    _current_user: AuthContextDep,
    prompt_config_id: Annotated[UUID, Path(description="LLM prompt config id.")],
    body: ProxyBody = None,
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "PATCH",
        f"/llm_prompt_configs/{prompt_config_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
        json_body=body,
    )
    return _upstream_response(status_code, payload)


@router.delete(
    "/guardrails/llm_prompt_configs/{prompt_config_id}",
    summary="Delete an LLM prompt config",
    description=load_description("guardrails/delete_llm_prompt_config.md"),
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses={200: DELETE_OK, **PROXY_ITEM_RESPONSES}),
)
def delete_guardrails_llm_prompt_config(
    _current_user: AuthContextDep,
    prompt_config_id: Annotated[UUID, Path(description="LLM prompt config id.")],
) -> Response:
    status_code, payload = proxy_guardrails_request(
        "DELETE",
        f"/llm_prompt_configs/{prompt_config_id}",
        organization_id=_current_user.organization_.id,
        project_id=_current_user.project_.id,
    )
    return _upstream_response(status_code, payload)


@router.get(
    "/guardrails/{job_id}",
    summary="Get guardrails job status",
    description=load_description("guardrails/get_guardrails_job.md"),
    response_model=APIResponse[GuardrailsJobPublic],
    dependencies=[Depends(require_permission(Permission.REQUIRE_PROJECT))],
    **doc(responses=JOB_ITEM_RESPONSES),
)
def get_guardrails_job_status(
    _current_user: AuthContextDep,
    session: SessionDep,
    job_id: Annotated[UUID, Path(description="Job id returned by POST /guardrails.")],
) -> APIResponse[GuardrailsJobPublic]:
    """Poll for a /guardrails job's status and result.

    On SUCCESS the sanitised text is rehydrated from the persisted upstream
    response stored on ``job.meta``.
    """
    project_id = _current_user.project_.id

    with log_context(
        tag="guardrails",
        system="guardrails",
        lifecycle="api.guardrails.status",
        job_id=str(job_id),
        project_id=project_id,
        organization_id=_current_user.organization_.id,
    ):
        job = JobCrud(session=session).get(job_id=job_id, project_id=project_id)
        if not job or job.job_type != JobType.LLM_GUARDRAILS:
            # 404 (not 403) to avoid leaking existence of non-guardrails jobs.
            raise HTTPException(status_code=404, detail="Job not found")

        meta = job.meta if isinstance(job.meta, dict) else {}
        callback_blob = meta.get("callback") if isinstance(meta, dict) else None

        warnings: list[str] = []
        if isinstance(callback_blob, dict):
            raw_warnings = callback_blob.get("warnings")
            if isinstance(raw_warnings, list):
                warnings = [w for w in raw_warnings if isinstance(w, str)]

        guardrails_response: GuardrailsCallbackData | None = None
        if job.status == JobStatus.SUCCESS:
            response_blob = meta.get("response") or {}
            data_blob = (
                response_blob.get("data") if isinstance(response_blob, dict) else None
            ) or {}
            safe_text = (
                data_blob.get("safe_text") if isinstance(data_blob, dict) else None
            )
            request_blob = meta.get("request") or {}
            original_text = (
                request_blob.get("text") if isinstance(request_blob, dict) else None
            )
            value = safe_text if isinstance(safe_text, str) else (original_text or "")

            response_id: str | None = None
            if isinstance(callback_blob, dict):
                rid = callback_blob.get("response_id")
                if isinstance(rid, str):
                    response_id = rid

            guardrails_response = GuardrailsCallbackData.model_validate(
                {
                    "response": {
                        "response_id": response_id,
                        "output": {
                            "type": "text",
                            "content": {"format": "text", "value": value},
                        },
                    },
                    "usage": (
                        data_blob.get("usage")
                        if isinstance(data_blob, dict)
                        and isinstance(data_blob.get("usage"), dict)
                        else {}
                    ),
                    "provider_raw_response": None,
                }
            )

        return APIResponse.success_response(
            data=GuardrailsJobPublic(
                job_id=job.id,
                status=job.status.value,
                guardrails_response=guardrails_response,
                error_message=job.error_message,
                warnings=warnings,
            )
        )
