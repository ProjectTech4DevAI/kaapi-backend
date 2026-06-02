"""L1 pipeline orchestrator.

Runs two filters in series for each row:
1. Topic Relevance (go/no-go) — REJECT stops the row.
2. Duplicate Detection (passthrough) — only on ACCEPTED rows.

"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from sqlmodel import Session

from app.core.batch.client import GeminiClient
from app.core.config import settings
from app.core.cloud import get_cloud_storage
from app.core.storage_utils import upload_jsonl_to_object_store
from app.models.assessment import AssessmentAttachment, AssessmentRun
from app.services.assessment.l1.duplicate_detection import run_duplicate_detection
from app.services.assessment.l1.topic_relevance import run_topic_relevance

logger = logging.getLogger(__name__)


def _build_l1_result(
    row_idx: int,
    tr_result: dict[str, Any] | None,
    dup_result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "row_id": f"row_{row_idx}",
        "l1_passed": tr_result["verdict"] if tr_result else True,
        "topic_relevance": {
            "decision": tr_result["decision"],
            "column_relevance": tr_result.get("column_relevance") or {},
            "reasoning": tr_result["reasoning"],
        }
        if tr_result
        else None,
        "duplicate_detection": dup_result,
    }


def run_l1_pipeline(
    run: AssessmentRun,
    rows: list[dict[str, str]],
    l1_config: dict[str, Any],
    session: Session,
    organization_id: int,
    project_id: int,
    attachments: list[AssessmentAttachment] | None = None,
) -> tuple[list[dict[str, str]], list[int], list[dict[str, Any]]]:
    """Run L1 filters on all rows.

    Args:
        run: The AssessmentRun record (used for S3 path and DB update).
        rows: Full dataset rows loaded from object store.
        l1_config: User-supplied config with topic_relevance and duplicate_detection keys.
        session: DB session.
        organization_id: For Gemini credential lookup.
        project_id: For Gemini credential lookup and S3 storage.

    Returns:
        (passed_rows, passed_indices, all_l1_results)
        passed_rows: subset of rows where topic_relevance verdict=true.
        passed_indices: original dataset indices of passed_rows (used to preserve row IDs in L2).
        all_l1_results: one entry per input row (len == len(rows)).
    """
    model = settings.ASSESSMENT_L1_GEMINI_MODEL
    workers = settings.ASSESSMENT_L1_CONCURRENT_WORKERS
    store_name = settings.ASSESSMENT_L1_DUPLICATE_STORE_NAME

    tr_config = l1_config.get("topic_relevance") or {}
    dup_config = l1_config.get("duplicate_detection") or {}

    tr_columns: list[str] = tr_config.get("columns") or []
    tr_prompt: str = tr_config.get("prompt") or ""
    dup_columns: list[str] = dup_config.get("columns") or []

    tr_attachment_columns = tr_config.get("attachment_columns")
    if tr_attachment_columns is None:
        tr_attachments = list(attachments or [])
    else:
        selected = set(tr_attachment_columns)
        tr_attachments = [a for a in (attachments or []) if a.column in selected]

    tr_enabled = bool(tr_columns and tr_prompt)
    dup_enabled = bool(dup_columns)

    if not tr_enabled and not dup_enabled:
        logger.warning(
            "[run_l1_pipeline] run_id=%s — no L1 filters configured, skipping L1",
            run.id,
        )
        return rows, list(range(len(rows))), []

    gemini_client = GeminiClient.from_credentials(
        session=session,
        org_id=organization_id,
        project_id=project_id,
    ).client

    logger.info(
        "[run_l1_pipeline] run_id=%s | rows=%s | model=%s | workers=%s | tr=%s | dup=%s",
        run.id,
        len(rows),
        model,
        workers,
        tr_enabled,
        dup_enabled,
    )

    # tr_results[idx] = None when TR disabled → no topic_relevance columns in export
    # Shared across rows so each unique attachment file is type-probed once.
    attachment_type_cache: dict[str, str] = {}

    tr_results: dict[int, dict[str, Any] | None] = {}
    if tr_enabled:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {
                executor.submit(
                    run_topic_relevance,
                    idx,
                    row,
                    tr_columns,
                    tr_prompt,
                    gemini_client,
                    model,
                    tr_attachments,
                    attachment_type_cache,
                ): idx
                for idx, row in enumerate(rows)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    tr_results[idx] = fut.result()
                except Exception as exc:
                    logger.warning(
                        "[run_l1_pipeline] TR future error row_%s | %s", idx, exc
                    )
                    tr_results[idx] = {
                        "row_id": f"row_{idx}",
                        "verdict": True,
                        "decision": "ACCEPT",
                        "column_relevance": {},
                        "reasoning": f"(future error — defaulting to pass) {exc}",
                    }
        passed_indices = [idx for idx, r in tr_results.items() if r and r["verdict"]]
    else:
        for idx in range(len(rows)):
            tr_results[idx] = None
        passed_indices = list(range(len(rows)))

    rejected_count = len(rows) - len(passed_indices)
    logger.info(
        "[run_l1_pipeline] run_id=%s | TR done | passed=%s | rejected=%s",
        run.id,
        len(passed_indices),
        rejected_count,
    )

    dup_results: dict[int, dict[str, Any]] = {}
    if dup_columns and passed_indices:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {
                executor.submit(
                    run_duplicate_detection,
                    idx,
                    rows[idx],
                    dup_columns,
                    gemini_client,
                    model,
                    store_name,
                ): idx
                for idx in passed_indices
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    dup_results[idx] = fut.result()
                except Exception as exc:
                    logger.warning(
                        "[run_l1_pipeline] DUP future error row_%s | %s", idx, exc
                    )
                    dup_results[idx] = {
                        "row_id": f"row_{idx}",
                        "verdict": "ERROR",
                        "match_title": None,
                        "source_url": None,
                        "matching_sentence": None,
                        "reason": str(exc)[:200],
                    }

    all_l1_results: list[dict[str, Any]] = [
        _build_l1_result(idx, tr_results[idx], dup_results.get(idx))
        for idx in range(len(rows))
    ]

    l1_object_store_url: str | None = None
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        l1_object_store_url = upload_jsonl_to_object_store(
            storage=storage,
            results=all_l1_results,
            filename="l1_results.json",
            subdirectory=f"assessment/run-{run.id}/l1",
            format="json",
        )
        logger.info(
            "[run_l1_pipeline] run_id=%s | L1 results uploaded to %s",
            run.id,
            l1_object_store_url,
        )
    except Exception as exc:
        logger.error(
            "[run_l1_pipeline] run_id=%s | S3 upload failed | %s",
            run.id,
            exc,
            exc_info=True,
        )

    from app.crud.assessment.core import update_assessment_run_l1_stats

    update_assessment_run_l1_stats(
        session=session,
        run=run,
        l1_object_store_url=l1_object_store_url,
        l1_total_rows=len(rows),
        l1_total_passed=len(passed_indices),
        l1_total_rejected=rejected_count,
    )

    sorted_passed_indices = sorted(passed_indices)
    passed_rows = [rows[idx] for idx in sorted_passed_indices]
    return passed_rows, sorted_passed_indices, all_l1_results
