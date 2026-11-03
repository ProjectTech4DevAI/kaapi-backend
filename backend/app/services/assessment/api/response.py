"""RESPONSE assessment: the pre-filter and assessment as direct provider calls on one row.

Requests, clients and parsing are the batch / llm-call ones; the result goes out by callback.
"""

import logging
from functools import partial
from http import HTTPStatus
from typing import Any
from uuid import UUID

from sqlmodel import Session

from app.core.db import engine
from app.crud.assessment import api
from app.crud.config import ConfigVersionCrud
from app.models.assessment import (
    Assessment,
    AssessmentOutput,
    AssessmentResult,
    AssessmentStatus,
    ParsedResult,
    PreFilter,
    PreFilterVerdict,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.models.llm.constants import DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS
from app.services.assessment.api.batch import (
    ApiStage,
    StageKind,
    build_pipeline,
    build_stage_requests,
    column_kinds,
    parse_one,
    parse_verdict,
    stage_params,
    stage_prompt,
    stage_provider_model,
)
from app.services.assessment.api.callbacks import deliver
from app.services.assessment.utils.parsing import parse_assessment_text
from app.services.llm.providers.registry import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


def _call_openai(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    response = client.responses.create(**request["body"])
    body = response.model_dump(mode="json")
    return {"response": {"status_code": HTTPStatus.OK, "body": body}}


def _call_anthropic(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    # The batch API injects max_tokens at batch level; a single call must carry it.
    params = {"max_tokens": DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS, **request["params"]}
    return {"response": client.messages.create(**params).model_dump(mode="json")}


def _call_google(client: Any, model: str, request: dict[str, Any]) -> dict[str, Any]:
    body = request["request"]
    config = dict(body.get("generationConfig", {}))
    if "systemInstruction" in body:
        config["systemInstruction"] = body["systemInstruction"]
    response = client.models.generate_content(
        model=model, contents=body["contents"], config=config
    )
    return {"response": response.model_dump(mode="json", by_alias=True)}


def _call_provider(
    provider_name: str, client: Any, model: str, request: dict[str, Any]
) -> dict[str, Any]:
    """Send one batch-shaped request; the raw response comes back in batch-result shape."""
    if provider_name == LLMProvider.OPENAI:
        return _call_openai(client, request)
    if provider_name == LLMProvider.ANTHROPIC:
        return _call_anthropic(client, request)
    if provider_name == LLMProvider.GOOGLE_GCP:
        return _call_google(client.genai, model, request)
    return _call_google(client, model, request)


def _run_stage(
    *,
    session: Session,
    blob: AssessmentConfigBlob,
    stage: str,
    row: dict[str, str],
    organization_id: int,
    project_id: int,
) -> ParsedResult:
    """One provider call for the row; a provider error comes back as the row's error."""
    provider_name, model = stage_provider_model(blob, stage)
    input_columns = blob.input_columns()
    text_columns, attachments = column_kinds(list(input_columns), input_columns)
    requests, _ = build_stage_requests(
        session=session,
        provider_name=provider_name,
        rows=[row],
        text_columns=text_columns,
        attachments=attachments,
        prompt=stage_prompt(blob, stage),
        params=stage_params(blob, stage),
        organization_id=organization_id,
        project_id=project_id,
    )
    client = get_llm_provider(
        session=session,
        provider_type=provider_name,
        project_id=project_id,
        organization_id=organization_id,
    ).client
    try:
        raw = _call_provider(provider_name, client, model, requests[0])
    except Exception as exc:
        raw = {"error": {"message": str(exc)}}
    return parse_one(raw, provider_name)


def _load_blob(
    session: Session, assessment: Assessment, config_id: UUID, config_version: int
) -> AssessmentConfigBlob:
    version = ConfigVersionCrud(
        session=session,
        config_id=config_id,
        project_id=assessment.project_id,
        tag=ConfigTag.ASSESSMENT,
    ).exists_or_raise(version_number=config_version)
    return AssessmentConfigBlob.model_validate(version.config_blob)


def _assess(
    *,
    session: Session,
    assessment: Assessment,
    config_id: UUID,
    config_version: int,
) -> AssessmentResult:
    blob = _load_blob(session, assessment, config_id, config_version)
    run_stage = partial(
        _run_stage,
        session=session,
        blob=blob,
        row=(assessment.input or {})["data"],
        organization_id=assessment.organization_id,
        project_id=assessment.project_id,
    )

    verdicts: dict[str, PreFilterVerdict] = {}
    gate_passed = True
    pre_filters = [
        step
        for step in build_pipeline(blob.pre_filters)
        if step["kind"] != StageKind.ASSESSMENT
    ]
    for step in pre_filters:
        verdict = parse_verdict(run_stage(stage=step["stage"])["output"])
        verdicts[step["stage"]] = PreFilterVerdict(**verdict)
        if step["kind"] == StageKind.GATE and not verdict["verdict"]:
            gate_passed = False

    pre_filter = PreFilter(**verdicts) if verdicts else None
    if not gate_passed:
        return AssessmentResult(output=AssessmentOutput(pre_filter=pre_filter))

    out = run_stage(stage=ApiStage.ASSESSMENT.value)
    return AssessmentResult(
        output=AssessmentOutput(
            assessment=parse_assessment_text(out), pre_filter=pre_filter
        ),
        error=out["error"],
    )


def run_response_assessment(
    *,
    assessment_id: UUID,
    config_id: UUID,
    config_version: int,
    callback_url: str,
    request_metadata: dict[str, Any] | None,
) -> None:
    """Assess one row, set the final status and deliver the result or the failure by callback."""
    with Session(engine) as session:
        assessment = session.get(Assessment, assessment_id)
        if assessment is None:
            logger.error(
                "[run_response_assessment] Assessment missing | assessment_id=%s",
                assessment_id,
            )
            return

        result: AssessmentResult | None = None
        failure: str | None = None
        try:
            result = _assess(
                session=session,
                assessment=assessment,
                config_id=config_id,
                config_version=config_version,
            )
        except Exception as exc:
            failure = str(exc)

        failed = failure or (result and result.error)
        api.update_status(
            session=session,
            obj=assessment,
            status=AssessmentStatus.FAILED if failed else AssessmentStatus.COMPLETED,
        )
        deliver(
            session=session,
            assessment=assessment,
            result=result,
            callback_url=callback_url,
            request_metadata=request_metadata,
            failure_message=failure,
        )
