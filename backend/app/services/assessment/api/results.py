"""Assemble the BATCH API-client result, for the webhook and the poll endpoint.

One result unit per input row. Gate-failed rows carry ``assessment=null`` plus
their pre-filter verdicts; gate-passed rows carry the assessment call's parsed
output. Both verdicts and outputs are streamed back from the stage dumps in object
storage; the execution bag holds nothing per row.
"""

import json
import logging
from typing import Any

from sqlmodel import Session

from app.crud.assessment import api
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.assessment import (
    ApiStage,
    Assessment,
    AssessmentBatchResult,
    AssessmentConfigRef,
    AssessmentCounts,
    AssessmentDetailResponse,
    AssessmentExecution,
    AssessmentOutput,
    AssessmentResult,
    AssessmentResultRow,
    AssessmentSubmission,
    AssessmentSummary,
    ParsedResult,
    PreFilter,
    Submission,
)
from app.services.assessment.api.batch import (
    _parse_verdict,
    load_stage_outputs,
    resolve_blob,
)

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


def _load_stage(
    session: Session,
    assessment: Assessment,
    blob: AssessmentConfigBlob,
    stage: ApiStage,
) -> tuple[dict[int, ParsedResult], str | None]:
    """One stage's dump as ``(outputs, load_error)``.

    ``load_error`` is set only on a failed read of a present dump, so the caller can
    flag it per-row instead of mistaking it for a clean run. No dump (stage not in the
    pipeline, or every row gated out) is a legit empty, not an error.
    """
    try:
        return load_stage_outputs(session, assessment, blob, stage) or {}, None
    except Exception as exc:
        logger.warning(
            "[_load_stage] Could not read stage output | assessment_id=%s | stage=%s | %s",
            assessment.id,
            stage,
            exc,
        )
        return {}, f"{stage.value} output could not be read from storage."


def build_result(
    *,
    session: Session,
    assessment: Assessment,
    bag: AssessmentExecution | None = None,
) -> AssessmentBatchResult:
    """Build the per-row result from the stage dumps: gate verdicts + assessment output.

    Status lives on the response envelope, so this body carries only rows and tallies.
    Pass ``bag`` when the caller already holds it, so a poll does not re-parse the column.
    """
    if bag is None:
        bag = api.load_execution_state(assessment)

    total_items = assessment.total_items
    blob = resolve_blob(session, assessment)

    tr_outputs: dict[int, ParsedResult] = {}
    tr_error: str | None = None
    rejected: set[int] = set()
    if bag and ApiStage.TOPIC_RELEVANCE in {step.stage for step in bag.pipeline}:
        tr_outputs, tr_error = _load_stage(
            session, assessment, blob, ApiStage.TOPIC_RELEVANCE
        )
        if ApiStage.TOPIC_RELEVANCE in bag.gate_stages():
            rejected = {
                idx
                for idx, out in tr_outputs.items()
                if not _parse_verdict(out.get("output")).verdict
            }

    outputs, load_error = _load_stage(session, assessment, blob, ApiStage.ASSESSMENT)
    # Rows the provider rejected are absent from the output dump; the bag kept their error.
    row_errors = bag.stage_errors.get(ApiStage.ASSESSMENT, {}) if bag else {}

    items: list[AssessmentResult] = []
    counts = AssessmentCounts()
    for idx in range(total_items):
        gate_passed = idx not in rejected
        # dict shape is the config's own json_output_schema (runtime-defined, no fixed
        # model); str for free-text output, None when gated/failed.
        assessment_output: dict[str, Any] | str | None = None
        error: str | None = None
        if gate_passed:
            if idx in outputs:
                out = outputs[idx]
                assessment_output = _parse_assessment(out)
                error = out.get("error")
            elif idx in row_errors:
                error = row_errors[idx]
            elif load_error or tr_error:
                error = load_error or tr_error

        tr_out = tr_outputs.get(idx)
        pre_filter = (
            PreFilter(topic_relevance=_parse_verdict(tr_out.get("output")))
            if tr_out
            else None
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
        if not gate_passed:
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
        open_submission_rows,
    )

    try:
        with open_submission_rows(session=session, assessment=assessment) as stream:
            return list(stream)
    except (SubmissionUnavailableError, ValueError) as exc:
        logger.warning(
            "[_submission_rows] Submission rows unavailable | assessment_id=%s | %s",
            assessment.id,
            exc,
        )
        return []


def build_summary(
    assessment: Assessment,
    submission_name: str | None = None,
    bag: AssessmentExecution | None = None,
) -> AssessmentSummary:
    """List row for one assessment. Reads only columns and the exec bag, never storage."""
    if bag is None:
        bag = api.load_execution_state(assessment)
    return AssessmentSummary(
        assessment_id=assessment.id,
        method=assessment.method,
        status=assessment.status,
        experiment_name=assessment.experiment_name,
        submission_id=assessment.submission_id,
        submission_name=submission_name,
        config=AssessmentConfigRef(
            id=assessment.config_id, version=assessment.config_version
        ),
        total_items=assessment.total_items,
        stages=[step.stage.value for step in bag.pipeline] if bag else [],
        stage=bag.stage.value if bag else None,
        stage_status=bag.stage_status.value if bag else None,
        error=bag.error if bag else None,
        inserted_at=assessment.inserted_at,
        updated_at=assessment.updated_at,
    )


def build_detail(
    *, session: Session, assessment: Assessment, include_input: bool = False
) -> AssessmentDetailResponse:
    """Poll payload: status plus every row produced so far.

    ``include_input`` echoes the submitted columns; it costs a storage read, so a tight
    poll loop leaves it off.
    """
    bag = api.load_execution_state(assessment)
    submission = (
        session.get(AssessmentSubmission, assessment.submission_id)
        if assessment.submission_id
        else None
    )

    result = build_result(session=session, assessment=assessment, bag=bag)
    rows = _submission_rows(session, assessment) if include_input else []

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
        assessment, submission.name if submission else None, bag=bag
    )
    return AssessmentDetailResponse(
        **summary.model_dump(),
        counts=result.counts,
        items=items,
    )
