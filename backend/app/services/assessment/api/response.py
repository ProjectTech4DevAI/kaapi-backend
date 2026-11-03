"""RESPONSE assessment: the pre-filter and assessment as direct provider calls on one row.

Requests, clients and parsing are the batch / llm-call ones; the run row records the stage
state and the result goes out by callback.
"""

import logging
from functools import partial
from http import HTTPStatus
from typing import Any, cast

from sqlmodel import Session

from app.core.db import engine
from app.crud.assessment import api
from app.models.assessment import (
    Assessment,
    AssessmentOutput,
    AssessmentResult,
    AssessmentRun,
    AssessmentStatus,
    BatchRunState,
    ParsedResult,
    PreFilter,
    PreFilterVerdict,
    StageStatus,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.llm.constants import DEFAULT_ASSESSMENT_BATCH_MAX_TOKENS
from app.services.assessment.api.batch import (
    ApiStage,
    StageKind,
    build_stage_requests,
    column_kinds,
    load_config_blob,
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

ROW = "0"


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


def _begin_stage(
    session: Session, execution: AssessmentRun, state: BatchRunState, stage: str
) -> None:
    state["stage"] = stage
    state["stage_status"] = StageStatus.PROCESSING.value
    api.save_execution_state(session=session, execution=execution, state=state)


def _record_error(state: BatchRunState, stage: str, error: str) -> None:
    state.setdefault("stage_errors", {})[stage] = {ROW: error}


def _assess(
    *,
    session: Session,
    assessment: Assessment,
    execution: AssessmentRun,
    state: BatchRunState,
    blob: AssessmentConfigBlob,
) -> AssessmentResult:
    run_stage = partial(
        _run_stage,
        session=session,
        blob=blob,
        row=(assessment.input or {})["data"],
        organization_id=assessment.organization_id,
        project_id=assessment.project_id,
    )

    verdicts: dict[str, PreFilterVerdict] = {}
    pre_filters = [
        step for step in state["pipeline"] if step["kind"] != StageKind.ASSESSMENT
    ]
    for step in pre_filters:
        stage = step["stage"]
        _begin_stage(session, execution, state, stage)
        out = run_stage(stage=stage)
        # A failed call fails open, as in BATCH; the error is still reported on the stage.
        verdict = parse_verdict(out["output"])
        verdicts[stage] = PreFilterVerdict(**verdict, error=out["error"])
        state.setdefault("verdicts", {})[stage] = {ROW: verdict}
        state.setdefault("counters", {})[stage] = {
            "total": 1,
            "passed": int(verdict["verdict"]),
            "rejected": int(not verdict["verdict"]),
        }
        if out["error"]:
            _record_error(state, stage, out["error"])
        elif step["kind"] == StageKind.GATE and not verdict["verdict"]:
            state["gate_passed"][0] = False
        state["stage_status"] = StageStatus.COMPLETED.value

    pre_filter = PreFilter(**verdicts) if verdicts else None
    if not state["gate_passed"][0]:
        return AssessmentResult(output=AssessmentOutput(pre_filter=pre_filter))

    _begin_stage(session, execution, state, ApiStage.ASSESSMENT.value)
    out = run_stage(stage=ApiStage.ASSESSMENT.value)
    if out["error"]:
        _record_error(state, ApiStage.ASSESSMENT.value, out["error"])
    state["stage_status"] = StageStatus.COMPLETED.value
    return AssessmentResult(
        output=AssessmentOutput(
            assessment=parse_assessment_text(out), pre_filter=pre_filter
        ),
        error=out["error"],
    )


def _outcome(
    state: BatchRunState, result: AssessmentResult | None, failure: str | None
) -> tuple[AssessmentStatus, str | None]:
    """Final status and the run's error message (stage-prefixed)."""
    if failure is not None:
        started = state["stage_status"] == StageStatus.PROCESSING.value
        message = f"{state['stage']}: {failure}" if started else failure
        return AssessmentStatus.FAILED, message

    errors = state.get("stage_errors") or {}
    message = (
        "; ".join(f"{stage}: {rows[ROW]}" for stage, rows in errors.items()) or None
    )
    if result and result.error:
        return AssessmentStatus.FAILED, message
    status = (
        AssessmentStatus.COMPLETED_WITH_ERRORS if errors else AssessmentStatus.COMPLETED
    )
    return status, message


def _finish(
    session: Session,
    assessment: Assessment,
    execution: AssessmentRun,
    state: BatchRunState,
    result: AssessmentResult | None,
    failure: str | None,
) -> None:
    status, message = _outcome(state, result, failure)
    if status == AssessmentStatus.FAILED:
        if state["stage_status"] != StageStatus.PENDING.value:
            state["stage_status"] = StageStatus.FAILED.value
        if message:
            state["error"] = message
    if result:
        state["result"] = result.model_dump(mode="json")
    api.save_execution_state(session=session, execution=execution, state=state)
    execution.error_message = message
    api.update_status(session=session, obj=execution, status=status)
    api.update_status(session=session, obj=assessment, status=status)


def run_response_assessment(*, execution_id: int) -> None:
    """Assess one row, record the run and deliver the result or the failure by callback."""
    with Session(engine) as session:
        execution = session.get(AssessmentRun, execution_id)
        assessment = (
            session.get(Assessment, execution.assessment_id) if execution else None
        )
        if execution is None or assessment is None:
            logger.error(
                "[run_response_assessment] Run missing | execution_id=%s", execution_id
            )
            return
        if execution.status in AssessmentStatus.terminal():
            return

        state = cast(BatchRunState, dict(execution.execution or {}))
        result: AssessmentResult | None = None
        failure: str | None = None
        try:
            blob = load_config_blob(session, execution, assessment.project_id)
            result = _assess(
                session=session,
                assessment=assessment,
                execution=execution,
                state=state,
                blob=blob,
            )
        except Exception as exc:
            failure = str(exc)

        _finish(session, assessment, execution, state, result, failure)
        deliver(
            session=session,
            assessment=assessment,
            result=result,
            callback_url=cast(str, state["callback_url"]),
            request_metadata=state["request_metadata"],
            failure_message=failure,
        )
