"""Assemble an AssessmentBatchResult for the BATCH API-client path.

One result unit per input row. Gate-failed rows carry ``assessment=null`` +
``metadata=null`` plus their pre-filter verdicts; gate-passed rows carry the
assessment call's parsed output + provider metadata. Pre-filter verdicts live in
the execution bag; assessment outputs are streamed back from object storage.
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
    AssessmentCounts,
    AssessmentMeta,
    AssessmentOutput,
    AssessmentResult,
    BatchInput,
    BatchRunState,
    ParsedResult,
    PreFilter,
    PreFilterVerdict,
    Verdict,
)
from app.models.llm.response import Usage
from app.services.assessment.api.batch import (
    ApiStage,
    build_rows,
    parse_batch_results,
)
from app.services.assessment.utils.parsing import parse_stored_results, usage_totals

logger = logging.getLogger(__name__)


def _to_usage(raw: dict[str, Any] | None) -> Usage | None:
    input_tokens, output_tokens, total_tokens = usage_totals(raw)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return Usage(
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        total_tokens=total_tokens or (input_tokens or 0) + (output_tokens or 0),
    )


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


def _to_meta(out: ParsedResult, provider: str, model: str) -> AssessmentMeta:
    return AssessmentMeta(
        provider=provider,
        model=model,
        response_id=out.get("response_id") or None,
        usage=_to_usage(out.get("usage")),
    )


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


def build_result(*, session: Session, assessment: Assessment) -> AssessmentBatchResult:
    """Build the per-row result from stored verdicts (bag) + assessment output (store).

    Status lives on the response envelope (sourced from the parent assessment), so this
    result body carries only the rows and their tallies.
    """
    executions = api.list_executions(session=session, assessment_id=assessment.id)
    bag = cast(BatchRunState, (executions[0].execution or {}) if executions else {})

    batch_input = (
        BatchInput.model_validate(assessment.input) if assessment.input else None
    )
    input_columns = bag.get("input_schema") or {}
    rows, _, _ = build_rows(batch_input, input_columns) if batch_input else ([], [], [])
    total_items = len(rows)

    gate_passed = bag.get("gate_passed") or [True] * total_items
    verdicts = bag.get("verdicts") or {}
    provider = bag.get("provider") or ""
    model = bag.get("model") or ""
    outputs, load_error = _load_assessment_outputs(session, bag, assessment.project_id)

    tr_verdicts = verdicts.get(ApiStage.TOPIC_RELEVANCE.value, {})
    dd_verdicts = verdicts.get(ApiStage.DUPLICATE_DETECTION.value, {})

    items: list[AssessmentResult] = []
    counts = AssessmentCounts()
    for idx in range(total_items):
        # dict shape is the config's own json_output_schema (runtime-defined, no fixed
        # model); str for free-text output, None when gated/failed.
        assessment_output: dict[str, Any] | str | None = None
        metadata: AssessmentMeta | None = None
        error: str | None = None
        if gate_passed[idx]:
            if idx in outputs:
                out = outputs[idx]
                assessment_output = _parse_assessment(out)
                metadata = _to_meta(out, provider, model)
                error = out.get("error")
            elif load_error:
                error = load_error

        topic_relevance = _verdict_obj(tr_verdicts.get(str(idx)))
        duplicate_detection = _verdict_obj(dd_verdicts.get(str(idx)))
        pre_filter = (
            PreFilter(
                topic_relevance=topic_relevance,
                duplicate_detection=duplicate_detection,
            )
            if (topic_relevance or duplicate_detection)
            else None
        )
        items.append(
            AssessmentResult(
                output=AssessmentOutput(
                    assessment=assessment_output,
                    pre_filter=pre_filter,
                ),
                metadata=metadata,
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
