"""Assemble the BATCH API-client result, for the webhook and the poll endpoint.

One result unit per input row. Gate-failed rows carry ``assessment=null`` plus
their pre-filter verdicts; gate-passed rows carry the assessment call's parsed
output. Pre-filter verdicts live in the execution bag; assessment outputs are
streamed back from object storage.
"""

import json
import logging
from typing import Any, cast

from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.crud.assessment import api
from app.models.assessment import (
    Assessment,
    AssessmentBatchResult,
    AssessmentConfigRef,
    AssessmentCounts,
    AssessmentDetailResponse,
    AssessmentOutput,
    AssessmentResult,
    AssessmentResultRow,
    AssessmentRun,
    AssessmentSubmission,
    AssessmentSummary,
    BatchRunState,
    ParsedResult,
    PreFilter,
    PreFilterVerdict,
    Submission,
    Verdict,
)
from app.services.assessment.api.batch import ApiStage, parse_batch_results
from app.services.assessment.utils.parsing import parse_stored_results

logger = logging.getLogger(__name__)


def _parse_assessment(out: ParsedResult) -> dict[str, Any] | str | None:
    """Stored assessment text as a dict (structured json_schema output) or raw string.

    Null when the row produced no assessment text (gated out or empty/failed call).
    """
    text = out.get("output")
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    return parsed if isinstance(parsed, dict) else text


def _verdict_obj(verdict: Verdict | None) -> PreFilterVerdict | None:
    if not verdict:
        return None
    return PreFilterVerdict(
        verdict=bool(verdict.get("verdict")),
        reasoning=str(verdict.get("reasoning") or ""),
    )


def _load_assessment_outputs(
    session: Session, bag: BatchRunState, project_id: int
) -> tuple[dict[int, ParsedResult], str | None]:
    """Stream + parse the assessment stage output. Returns ``(outputs, load_error)``.

    ``load_error`` is set only on a failed read of a present URL, so the caller
    can flag it per-row instead of mistaking it for a clean run. Missing URL
    (all rows gated) is a legit empty, not an error.
    """
    url = (bag.get("stage_output_urls") or {}).get(ApiStage.ASSESSMENT.value)
    if not url:
        return {}, None
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        raw = parse_stored_results(storage.stream(url).read().decode("utf-8"))
        return parse_batch_results(raw, bag.get("provider")), None
    except Exception as exc:
        logger.warning(
            "[_load_assessment_outputs] Could not read assessment output | url=%s | %s",
            url,
            exc,
        )
        return {}, "Assessment output could not be read from storage."


def build_result(
    *,
    session: Session,
    assessment: Assessment,
    execution: AssessmentRun | None = None,
) -> AssessmentBatchResult:
    """Build the per-row result from stored verdicts (bag) + assessment output (store).

    Status lives on the response envelope (sourced from the parent assessment), so this
    result body carries only the rows and their tallies. Pass ``execution`` when the
    caller already holds it, so a poll does not re-query it.
    """
    if execution is None:
        executions = api.list_executions(session=session, assessment_id=assessment.id)
        execution = executions[0] if executions else None
    bag = cast(BatchRunState, (execution.execution or {}) if execution else {})

    # Off the execution, not re-derived: the terminal path must not fetch rows to count them.
    total_items = execution.total_items if execution else 0

    gate_passed = bag.get("gate_passed") or [True] * total_items
    verdicts = bag.get("verdicts") or {}
    outputs, load_error = _load_assessment_outputs(session, bag, assessment.project_id)

    tr_verdicts = verdicts.get(ApiStage.TOPIC_RELEVANCE.value, {})

    items: list[AssessmentResult] = []
    counts = AssessmentCounts()
    for idx in range(total_items):
        # dict shape is the config's own json_output_schema (runtime-defined, no fixed
        # model); str for free-text output, None when gated/failed.
        assessment_output: dict[str, Any] | str | None = None
        error: str | None = None
        if gate_passed[idx]:
            if idx in outputs:
                out = outputs[idx]
                assessment_output = _parse_assessment(out)
                error = out.get("error")
            elif load_error:
                error = load_error

        topic_relevance = _verdict_obj(tr_verdicts.get(str(idx)))
        pre_filter = (
            PreFilter(topic_relevance=topic_relevance) if topic_relevance else None
        )
        items.append(
            AssessmentResult(
                output=AssessmentOutput(
                    assessment=assessment_output,
                    pre_filter=pre_filter,
                ),
                error=error,
            )
        )

        if assessment_output is not None:
            counts.assessed += 1
        if not gate_passed[idx]:
            counts.filtered += 1
        if error:
            counts.errors += 1

    return AssessmentBatchResult(
        total_items=total_items,
        counts=counts,
        items=items,
    )


def _submission_rows(session: Session, assessment: Assessment) -> list[Submission]:
    """The rows the run was submitted with; empty when they cannot be read.

    A storage blip must degrade the poll to rows without input columns, never fail it.
    """
    from app.services.assessment.api.submission_store import (
        SubmissionUnavailableError,
        load_submission_rows,
    )

    try:
        return load_submission_rows(session=session, assessment=assessment).data or []
    except (SubmissionUnavailableError, ValueError) as exc:
        logger.warning(
            "[_submission_rows] Submission rows unavailable | assessment_id=%s | %s",
            assessment.id,
            exc,
        )
        return []


def build_summary(
    assessment: Assessment,
    execution: AssessmentRun | None,
    submission_name: str | None = None,
) -> AssessmentSummary:
    """List row for one assessment. Reads only columns and the exec bag, never storage."""
    bag = cast(BatchRunState, (execution.execution or {}) if execution else {})
    config = (
        AssessmentConfigRef(id=execution.config_id, version=execution.config_version)
        if execution
        else None
    )
    return AssessmentSummary(
        assessment_id=assessment.id,
        method=assessment.method,
        status=assessment.status,
        experiment_name=assessment.experiment_name,
        submission_id=assessment.submission_id,
        submission_name=submission_name,
        config=config,
        total_items=execution.total_items if execution else 0,
        stages=[
            str(step["stage"]) for step in bag.get("pipeline", []) if step.get("stage")
        ],
        stage=bag.get("stage"),
        stage_status=bag.get("stage_status"),
        error=execution.error_message if execution else None,
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
    )


def build_detail(
    *, session: Session, assessment: Assessment
) -> AssessmentDetailResponse:
    """Poll payload: run status plus every row produced so far, input columns attached."""
    executions = api.list_executions(session=session, assessment_id=assessment.id)
    execution = executions[0] if executions else None
    submission = (
        session.get(AssessmentSubmission, assessment.submission_id)
        if assessment.submission_id
        else None
    )

    result = build_result(session=session, assessment=assessment, execution=execution)
    rows = _submission_rows(session, assessment)

    items = [
        AssessmentResultRow(
            row_index=idx,
            input=rows[idx] if idx < len(rows) else None,
            output=item.output,
            error=item.error,
        )
        for idx, item in enumerate(result.items)
    ]

    summary = build_summary(
        assessment, execution, submission.name if submission else None
    )
    return AssessmentDetailResponse(
        **summary.model_dump(),
        counts=result.counts,
        items=items,
    )
