"""Fast evaluation orchestration (run_mode="fast").

Synchronous text-eval path: makes Responses + Embeddings calls in parallel from
a single Celery task and persists per-stage units to S3. Each stage is skipped
on retry if its `batch_job` row already exists.

    Stage 1 — Responses unit:   evaluation_run.batch_job_id
    Stage 2 — Embeddings unit:  evaluation_run.embedding_batch_job_id
    Stage 3 — Score + trace + cost (no marker; each step is idempotent)
    Stage 4 — Mark completed
    Stage 5 — Persist score unit (summary + per-trace) via the shared
              save_score helper, so the cached trace unit (score_trace_url)
              exists immediately and the read path (trace view / resync /
              grouped export) mirrors the batch path without racing Langfuse
              ingestion.

See `Fast Evaluation SRD.md` for the full design.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import openai
from langfuse import Langfuse
from openai import OpenAI
from pydantic import ValidationError
from sqlalchemy import Integer
from sqlmodel import Session, select
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.cloud.storage import get_cloud_storage
from app.core.config import settings
from app.core.storage_utils import (
    load_json_from_object_store,
    upload_jsonl_to_object_store,
)
from app.crud.evaluations.core import (
    resolve_evaluation_config,
    resolve_model_from_config,
    save_score,
    update_evaluation_run,
)
from app.crud.evaluations.cost import attach_cost
from app.crud.evaluations.embeddings import (
    EMBEDDING_MODEL,
    calculate_cosine_similarity,
)
from app.crud.evaluations.judge import (
    JUDGE_COST_STAGE,
    JudgeInputEnum,
    JudgeMetricEnum,
    JudgeMetricSpec,
    JudgeResult,
    build_judge_params,
    enabled_metric_specs,
    judge_row,
)
from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
)
from app.crud.evaluations.merge import apply_cosine_breakdown
from app.crud.evaluations.score import (
    COSINE_SCORE_COMMENT,
    COSINE_SCORE_NAME,
    JUDGE_FAILED_REASON,
    EvaluationScore,
    TraceData,
    TraceScore,
)
from app.crud.job import (
    create_batch_job,
    delete_batch_job,
    get_batch_job,
)
from app.models import EvaluationRun, EvaluationRunUpdate
from app.models.batch_job import BatchJob, BatchJobCreate
from app.models.evaluation import RunModeEnum
from app.models.llm.request import TextLLMParams
from app.services.llm.mappers import map_kaapi_to_openai_params
from app.services.response.response import get_file_search_results

logger = logging.getLogger(__name__)


# job_type discriminators on batch_job for the two fast-path stages. The row's
# presence + raw_output_url is what marks a stage as already done on retry.
JOB_TYPE_EVALUATION_FAST = "evaluation_fast"
JOB_TYPE_EVALUATION_FAST_CHUNK = "evaluation_fast_chunk"
JOB_TYPE_EMBEDDING_FAST = "embedding_fast"

# batch_job.config keys tying a chunk row back to its run + slice.
CHUNK_CONFIG_RUN_ID = "eval_run_id"
CHUNK_CONFIG_INDEX = "chunk_index"

# Reasons a row cannot be scored. embedding_failed is v1-only
# (cosine); v2 judged runs never embed, so only the empty-side reasons apply.
UNSCOREABLE_EMPTY_OUTPUT = "empty_output"
UNSCOREABLE_EMPTY_GROUND_TRUTH = "empty_ground_truth"
UNSCOREABLE_EMBEDDING_FAILED = "embedding_failed"

# Judge tell the template apart from the instructions above it.
PROMPT_TEMPLATE_LABEL = "Prompt template wrapped around each user input:"

# How many top KB matches to name in the knowledge_base trace comment.
_KB_TOP_CHUNKS = 3


def _format_top_kb_matches(sorted_chunks: list[dict[str, Any]]) -> str:
    """Top-N retrieved chunks as 'biu-1.pdf (90.6%), faq.pdf (66.3%)'.

    Expects chunks pre-sorted by score desc; old S3 payloads may lack filename.
    """
    matches = [
        f"{c.get('filename') or 'unknown'} ({c.get('score', 0) * 100:.1f}%)"
        for c in sorted_chunks
    ]
    return ", ".join(matches[:_KB_TOP_CHUNKS])


# Per-call retry policy for Stage 1 / Stage 2.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 30.0

_RETRYABLE_OPENAI_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


# reraise=True so call-site handlers see the original OpenAIError, not RetryError.
_retry_openai_call = retry(
    retry=retry_if_exception_type(_RETRYABLE_OPENAI_ERRORS),
    wait=wait_random_exponential(
        multiplier=_RETRY_BASE_DELAY_SECONDS, max=_RETRY_MAX_DELAY_SECONDS
    ),
    stop=stop_after_attempt(_RETRY_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)


@_retry_openai_call
def _create_response(openai_client: OpenAI, params: dict[str, Any]) -> Any:
    return openai_client.responses.create(**params)


@_retry_openai_call
def _create_embedding(
    openai_client: OpenAI, *, model: str, output_text: str, ground_truth: str
) -> Any:
    return openai_client.embeddings.create(
        model=model, input=[output_text, ground_truth], encoding_format="float"
    )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an object or dict (SDK object vs test dict), with a default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_response_text(response: Any) -> str:
    """Extract generated text, preferring `output_text` then walking `output`."""
    output_text = _field(response, "output_text")
    if output_text:
        return output_text

    output = _field(response, "output")
    if not output:
        return ""

    for item in output:
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content") or []:
            if _field(content, "type") == "output_text":
                text = _field(content, "text")
                if text:
                    return text
    return ""


def _response_result(
    *,
    item_id: str,
    question: str,
    ground_truth: str,
    question_id: Any,
    generated_output: str,
    failed: bool,
    response_id: str | None = None,
    usage: dict[str, int] | None = None,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One Stage-1 per-item result, in the batch path's shape."""
    return {
        "item_id": item_id,
        "question": question,
        "generated_output": generated_output,
        "ground_truth": ground_truth,
        "response_id": response_id,
        "usage": usage,
        "question_id": question_id,
        "failed": failed,
        "retrieved_chunks": retrieved_chunks,
    }


def _responses_call_for_item(
    *,
    openai_client: OpenAI,
    base_params: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    """Run one Responses call for a dataset item, in the batch path's per-item shape.

    `base_params` is the question-independent OpenAI body produced once by
    `map_kaapi_to_openai_params`; only `input` varies per item.
    """
    item_id = item["id"]
    question = item["input"].get("question", "") if item.get("input") else ""
    ground_truth = (
        item["expected_output"].get("answer", "") if item.get("expected_output") else ""
    )
    question_id = (item.get("metadata") or {}).get("question_id")

    if not question:
        return _response_result(
            item_id=item_id,
            question="",
            ground_truth=ground_truth,
            question_id=question_id,
            generated_output="ERROR: missing question in dataset item",
            failed=True,
        )

    params = {**base_params, "input": question}

    try:
        response = _create_response(openai_client, params)
    except openai.OpenAIError as exc:
        logger.warning(
            f"[_responses_call_for_item] Item failed | item_id={item_id} | error={exc}"
        )
        return _response_result(
            item_id=item_id,
            question=question,
            ground_truth=ground_truth,
            question_id=question_id,
            generated_output=f"ERROR: {exc}",
            failed=True,
        )

    usage = getattr(response, "usage", None)
    return _response_result(
        item_id=item_id,
        question=question,
        ground_truth=ground_truth,
        question_id=question_id,
        generated_output=_extract_response_text(response),
        response_id=getattr(response, "id", None),
        usage={
            "input_tokens": int(_field(usage, "input_tokens", 0) or 0),
            "output_tokens": int(_field(usage, "output_tokens", 0) or 0),
            "total_tokens": int(_field(usage, "total_tokens", 0) or 0),
        },
        failed=False,
        # Plain dicts (not FileResultChunk) so the unit stays JSON-serializable for S3.
        retrieved_chunks=[
            {"score": c.score, "text": c.text, "filename": c.filename}
            for c in get_file_search_results(response)
        ],
    )


def _embedding_failure(item_id: str, error: str) -> dict[str, Any]:
    """One failed Stage-2 per-pair result."""
    return {
        "item_id": item_id,
        "output_embedding": None,
        "ground_truth_embedding": None,
        "usage": None,
        "failed": True,
        "error": error,
    }


def _embedding_call_for_pair(
    *,
    openai_client: OpenAI,
    embedding_model: str,
    item_id: str,
    output_text: str,
    ground_truth: str,
) -> dict[str, Any]:
    """Embed an (output, ground_truth) pair; `failed=True` on a terminal failure."""
    if not output_text or not ground_truth:
        return _embedding_failure(item_id, "empty output or ground_truth")

    try:
        response = _create_embedding(
            openai_client,
            model=embedding_model,
            output_text=output_text,
            ground_truth=ground_truth,
        )
    except openai.OpenAIError as exc:
        logger.warning(
            f"[_embedding_call_for_pair] Item failed | item_id={item_id} | error={exc}"
        )
        return _embedding_failure(item_id, str(exc))

    data = _field(response, "data") or []
    if len(data) < 2:
        return _embedding_failure(item_id, f"expected 2 embeddings, got {len(data)}")

    output_embedding: list[float] | None = None
    ground_truth_embedding: list[float] | None = None
    for emb in data:
        index = _field(emb, "index")
        vector = _field(emb, "embedding")
        if index == 0:
            output_embedding = vector
        elif index == 1:
            ground_truth_embedding = vector

    usage_obj = _field(response, "usage")
    usage_dict: dict[str, int] = {
        "prompt_tokens": int(_field(usage_obj, "prompt_tokens", 0) or 0),
        "total_tokens": int(_field(usage_obj, "total_tokens", 0) or 0),
    }

    return {
        "item_id": item_id,
        "output_embedding": output_embedding,
        "ground_truth_embedding": ground_truth_embedding,
        "usage": usage_dict,
        "failed": output_embedding is None or ground_truth_embedding is None,
    }


def _is_failure_threshold_breached(*, failed_rows: int, total_rows: int) -> bool:
    """True if the failed-row fraction exceeds EVAL_FAST_FAILURE_THRESHOLD."""
    if total_rows == 0:
        return False
    return (failed_rows / total_rows) > settings.EVAL_FAST_FAILURE_THRESHOLD


def _sum_usage(results: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    """Sum the per-item `usage` token counts across results, for the given keys."""
    totals = dict.fromkeys(keys, 0)
    for r in results:
        usage = r.get("usage") or {}
        for k in keys:
            totals[k] += int(usage.get(k, 0) or 0)
    return totals


def _upload_unit_to_s3(
    *,
    session: Session,
    project_id: int,
    eval_run_id: int,
    filename: str,
    results: list[dict[str, Any]],
) -> str | None:
    """Upload a stage unit (responses or embeddings) as JSON to S3."""
    storage = get_cloud_storage(session=session, project_id=project_id)
    return upload_jsonl_to_object_store(
        storage=storage,
        results=results,
        filename=filename,
        subdirectory=f"evaluations/fast/{eval_run_id}",
        format="json",
    )


def _load_unit_from_s3(
    *, session: Session, project_id: int, url: str
) -> list[dict[str, Any]]:
    """Load a stage unit back from S3. Raises if the unit cannot be loaded."""
    storage = get_cloud_storage(session=session, project_id=project_id)
    data = load_json_from_object_store(storage=storage, url=url)
    if data is None:
        raise RuntimeError(f"Failed to load fast eval unit from S3 | url={url}")
    if not isinstance(data, list):
        raise RuntimeError(
            f"Fast eval unit at {url} is not a list | type={type(data).__name__}"
        )
    return data


def _load_completed_stage(
    *,
    session: Session,
    batch_job_id: int | None,
    project_id: int,
    log_prefix: str,
    stage: str,
) -> list[dict[str, Any]] | None:
    """Return a stage's persisted unit if its batch_job already completed, else None.

    This is the per-stage retry skip: a `batch_job` row with a `raw_output_url`
    means the stage finished on an earlier attempt, so we reload from S3 instead
    of re-calling OpenAI.
    """
    if not batch_job_id:
        return None
    existing = get_batch_job(session=session, batch_job_id=batch_job_id)
    if not (existing and existing.raw_output_url):
        return None
    logger.info(
        f"[{stage}] {log_prefix} Skipping (already done) | batch_job_id={existing.id}"
    )
    return _load_unit_from_s3(
        session=session, project_id=project_id, url=existing.raw_output_url
    )


def list_response_chunk_jobs(*, session: Session, eval_run_id: int) -> list[BatchJob]:
    """All response-chunk batch_jobs for a fast run, in any state."""
    statement = select(BatchJob).where(
        BatchJob.job_type == JOB_TYPE_EVALUATION_FAST_CHUNK,
        BatchJob.config[CHUNK_CONFIG_RUN_ID].astext.cast(Integer) == eval_run_id,
    )
    return list(session.exec(statement).all())


def _get_chunk_job(
    *, session: Session, eval_run_id: int, chunk_index: int
) -> BatchJob | None:
    """The chunk batch_job for one (eval_run, chunk_index), or None."""
    statement = select(BatchJob).where(
        BatchJob.job_type == JOB_TYPE_EVALUATION_FAST_CHUNK,
        BatchJob.config[CHUNK_CONFIG_RUN_ID].astext.cast(Integer) == eval_run_id,
        BatchJob.config[CHUNK_CONFIG_INDEX].astext.cast(Integer) == chunk_index,
    )
    return session.exec(statement).first()


def _cleanup_response_chunks(*, session: Session, eval_run: EvaluationRun) -> None:
    """Delete the per-chunk S3 files + batch_job rows once a run completes.

    Best-effort: a failed delete only leaks DB+S3 bloat, so it never fails the
    run. Failed runs skip this and keep their chunks for the healer.
    """
    try:
        storage = get_cloud_storage(session=session, project_id=eval_run.project_id)
        chunk_jobs = list_response_chunk_jobs(session=session, eval_run_id=eval_run.id)
        for job in chunk_jobs:
            if job.raw_output_url:
                storage.delete(job.raw_output_url)
            delete_batch_job(session, job)
        logger.info(
            f"[_cleanup_response_chunks] Removed {len(chunk_jobs)} chunk "
            f"artifacts | eval_run_id={eval_run.id}"
        )
    except Exception as exc:
        logger.warning(
            f"[_cleanup_response_chunks] Cleanup failed (orphans harmless) | "
            f"eval_run_id={eval_run.id} | error={exc}",
            exc_info=True,
        )


def run_response_chunk(
    *,
    session: Session,
    openai_client: OpenAI,
    eval_run: EvaluationRun,
    config: TextLLMParams,
    dataset_items_slice: list[dict[str, Any]],
    chunk_index: int,
    log_prefix: str,
) -> None:
    """Run the responses stage over one slice of dataset items.

    Idempotent: skipped when this (eval_run, chunk_index) already has a
    raw_output_url, so redelivery or a cron re-enqueue never re-charges OpenAI.
    The failure threshold is not checked here — it's decided over the merged set
    at aggregation.

    Concurrency: two workers racing the same chunk_index can both pass the skip
    guard and write two rows; the merge de-duplicates per index.
    """
    existing = _get_chunk_job(
        session=session, eval_run_id=eval_run.id, chunk_index=chunk_index
    )
    if existing and existing.raw_output_url:
        logger.info(
            f"[run_response_chunk] {log_prefix} Skipping chunk (already done) | "
            f"chunk_index={chunk_index} | batch_job_id={existing.id}"
        )
        return

    logger.info(
        f"[run_response_chunk] {log_prefix} Running chunk | "
        f"chunk_index={chunk_index} | items={len(dataset_items_slice)} | "
        f"model={config.model} | concurrency={settings.EVAL_FAST_API_CONCURRENCY}"
    )

    base_params, mapper_warnings = map_kaapi_to_openai_params(
        session=session, kaapi_params=config.model_dump(exclude_unset=True)
    )
    if mapper_warnings:
        logger.info(
            f"[run_response_chunk] {log_prefix} Mapper warnings: {mapper_warnings}"
        )

    # Ask OpenAI to return the file_search hits so knowledge_base can judge them.
    # tool_choice stays at the model default (auto) — consistent with normal calls;
    # a row where the model doesn't query the KB is scored N/A, not forced to search.
    if any(t.get("type") == "file_search" for t in base_params.get("tools", [])):
        base_params["include"] = ["file_search_call.results"]

    results: list[dict[str, Any]] = []
    max_workers = max(
        1, min(settings.EVAL_FAST_API_CONCURRENCY, len(dataset_items_slice))
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _responses_call_for_item,
                openai_client=openai_client,
                base_params=base_params,
                item=item,
            ): item["id"]
            for item in dataset_items_slice
        }
        for future in as_completed(futures):
            results.append(future.result())

    failed_count = sum(1 for r in results if r.get("failed"))
    logger.info(
        f"[run_response_chunk] {log_prefix} Chunk finished | "
        f"chunk_index={chunk_index} | total={len(results)} | failed={failed_count}"
    )

    raw_output_url = _upload_unit_to_s3(
        session=session,
        project_id=eval_run.project_id,
        eval_run_id=eval_run.id,
        filename=f"responses_{eval_run.id}_{chunk_index}.json",
        results=results,
    )

    summed_usage = _sum_usage(
        results, ("input_tokens", "output_tokens", "total_tokens")
    )
    create_batch_job(
        session=session,
        batch_job_create=BatchJobCreate(
            provider="openai",
            job_type=JOB_TYPE_EVALUATION_FAST_CHUNK,
            config={
                "endpoint": "/v1/responses",
                "run_mode": RunModeEnum.FAST.value,
                "model": config.model,
                "usage": summed_usage,
                CHUNK_CONFIG_RUN_ID: eval_run.id,
                CHUNK_CONFIG_INDEX: chunk_index,
            },
            raw_output_url=raw_output_url,
            total_items=len(results),
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
        ),
    )


def _merge_response_chunks(
    *,
    session: Session,
    eval_run: EvaluationRun,
) -> tuple[EvaluationRun, list[dict[str, Any]]]:
    """Concatenate every response chunk into the canonical responses unit.

    Skipped on retry when `eval_run.batch_job_id` is set (canonical unit
    reloaded from S3) so aggregate redelivery never re-merges. Chunks are
    ordered by index and de-duplicated per index — a healer re-enqueue may race
    a slow chunk — so the merged order, and the scores, stay reproducible.
    """
    log_prefix = (
        f"[org={eval_run.organization_id}]"
        f"[project={eval_run.project_id}]"
        f"[eval={eval_run.id}]"
    )
    cached = _load_completed_stage(
        session=session,
        batch_job_id=eval_run.batch_job_id,
        project_id=eval_run.project_id,
        log_prefix=log_prefix,
        stage="_merge_response_chunks",
    )
    if cached is not None:
        return eval_run, cached

    chunk_jobs = list_response_chunk_jobs(session=session, eval_run_id=eval_run.id)
    chunk_job_by_index: dict[int, BatchJob] = {}
    for job in chunk_jobs:
        chunk_index = int(job.config.get(CHUNK_CONFIG_INDEX, -1))
        if job.raw_output_url and chunk_index not in chunk_job_by_index:
            chunk_job_by_index[chunk_index] = job

    results: list[dict[str, Any]] = []
    for chunk_index in sorted(chunk_job_by_index):
        results.extend(
            _load_unit_from_s3(
                session=session,
                project_id=eval_run.project_id,
                url=chunk_job_by_index[chunk_index].raw_output_url,
            )
        )

    logger.info(
        f"[_merge_response_chunks] {log_prefix} Merged chunks | "
        f"chunks={len(chunk_job_by_index)} | items={len(results)}"
    )

    raw_output_url = _upload_unit_to_s3(
        session=session,
        project_id=eval_run.project_id,
        eval_run_id=eval_run.id,
        filename=f"responses_{eval_run.id}.json",
        results=results,
    )

    model = (
        next(iter(chunk_job_by_index.values())).config.get("model")
        if chunk_job_by_index
        else None
    )
    summed_usage = _sum_usage(
        results, ("input_tokens", "output_tokens", "total_tokens")
    )
    batch_job = create_batch_job(
        session=session,
        batch_job_create=BatchJobCreate(
            provider="openai",
            job_type=JOB_TYPE_EVALUATION_FAST,
            config={
                "endpoint": "/v1/responses",
                "run_mode": RunModeEnum.FAST.value,
                "model": model,
                "usage": summed_usage,
            },
            raw_output_url=raw_output_url,
            total_items=len(results),
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
        ),
    )

    # batch_job_id / total_items aren't on EvaluationRunUpdate; set them directly.
    eval_run.batch_job_id = batch_job.id
    eval_run.total_items = len(results)
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(),
    )

    return eval_run, results


def _stage2_embeddings(
    *,
    session: Session,
    openai_client: OpenAI,
    eval_run: EvaluationRun,
    response_results: list[dict[str, Any]],
    log_prefix: str,
) -> tuple[EvaluationRun, list[dict[str, Any]]]:
    """Stage 2 — embed each (output, ground_truth) pair; skipped on retry if done."""
    cached = _load_completed_stage(
        session=session,
        batch_job_id=eval_run.embedding_batch_job_id,
        project_id=eval_run.project_id,
        log_prefix=log_prefix,
        stage="_stage2_embeddings",
    )
    if cached is not None:
        return eval_run, cached

    # Only embed items that succeeded in Stage 1.
    embed_candidates = [r for r in response_results if not r.get("failed")]
    logger.info(
        f"[_stage2_embeddings] {log_prefix} Running stage 2 | "
        f"items={len(embed_candidates)} | model={EMBEDDING_MODEL} | "
        f"concurrency={settings.EVAL_FAST_API_CONCURRENCY}"
    )

    embedding_results: list[dict[str, Any]] = []
    max_workers = max(
        1, min(settings.EVAL_FAST_API_CONCURRENCY, len(embed_candidates) or 1)
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _embedding_call_for_pair,
                openai_client=openai_client,
                embedding_model=EMBEDDING_MODEL,
                item_id=r["item_id"],
                output_text=r.get("generated_output", ""),
                ground_truth=r.get("ground_truth", ""),
            ): r["item_id"]
            for r in embed_candidates
        }
        for future in as_completed(futures):
            embedding_results.append(future.result())

    failed_count = sum(1 for r in embedding_results if r.get("failed"))
    # Threshold is over the whole dataset: Stage 1 failures count as failures too.
    total_failures = failed_count + sum(1 for r in response_results if r.get("failed"))
    logger.info(
        f"[_stage2_embeddings] {log_prefix} Stage 2 finished | "
        f"total={len(embedding_results)} | failed={failed_count}"
    )

    if _is_failure_threshold_breached(
        failed_rows=total_failures, total_rows=len(response_results)
    ):
        raise RuntimeError(
            f"Fast eval Stage 2 exceeded failure threshold | "
            f"failed={total_failures}/{len(response_results)} | "
            f"threshold={settings.EVAL_FAST_FAILURE_THRESHOLD}"
        )

    raw_output_url = _upload_unit_to_s3(
        session=session,
        project_id=eval_run.project_id,
        eval_run_id=eval_run.id,
        filename=f"embeddings_{eval_run.id}.json",
        results=embedding_results,
    )

    summed_usage = _sum_usage(embedding_results, ("prompt_tokens", "total_tokens"))

    batch_job = create_batch_job(
        session=session,
        batch_job_create=BatchJobCreate(
            provider="openai",
            job_type=JOB_TYPE_EMBEDDING_FAST,
            config={
                "endpoint": "/v1/embeddings",
                "run_mode": RunModeEnum.FAST.value,
                "embedding_model": EMBEDDING_MODEL,
                "usage": summed_usage,
            },
            raw_output_url=raw_output_url,
            total_items=len(embedding_results),
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
        ),
    )

    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(embedding_batch_job_id=batch_job.id),
    )

    return eval_run, embedding_results


def _resolve_config_prompt(
    *, session: Session, eval_run: EvaluationRun, log_prefix: str
) -> str | None:
    """The evaluated bot's own configured prompt, or None if unresolvable.

    The prompt template is appended when the config carries one, since it is equally
    part of what the bot was told to do. Returns None when the config carries no
    instructions, so the caller drops the prompt metric rather than grading against "".
    """
    if not eval_run.config_id or not eval_run.config_version:
        return None

    config, error = resolve_evaluation_config(
        session=session,
        config_id=eval_run.config_id,
        config_version=eval_run.config_version,
        project_id=eval_run.project_id,
    )
    if error or config is None:
        return None

    # Native/proxy params aren't TextLLMParams-shaped; a mismatch just means there are
    # no instructions to grade against, not a run failure.
    try:
        params = TextLLMParams.model_validate(config.completion.params)
    except ValidationError as exc:
        logger.info(
            f"[_resolve_config_prompt] {log_prefix} Completion params are not "
            f"text params; prompt metric unscoreable | error={exc}"
        )
        return None

    sections: list[str] = []
    if params.instructions:
        sections.append(params.instructions.strip())
    if config.prompt_template and config.prompt_template.template:
        sections.append(
            f"{PROMPT_TEMPLATE_LABEL}\n{config.prompt_template.template.strip()}"
        )

    if not sections:
        return None
    return "\n\n".join(sections)


def _judge_rows(
    *,
    session: Session,
    openai_client: OpenAI,
    metrics: list[JudgeMetricSpec],
    config_prompt: str,
    judgeable: list[tuple[str, str, dict[str, Any]]],
    log_prefix: str,
) -> tuple[dict[str, JudgeResult], set[str], str | None]:
    """Run one combined judge completion per judgeable row, isolated per row.

    `metrics` is the run's enabled set, already filtered for unresolvable run-level
    inputs; `config_prompt` is the same run-level text for every row.
    """
    results: dict[str, JudgeResult] = {}
    failed_refs: set[str] = set()
    if not judgeable or not metrics:
        return results, failed_refs, None

    # Build base params once per run; judging is system-config only, so every metric
    # uses its built-in prompt + shared model. Instructions vary per row (by
    # applicable-metric subset) and are composed inside judge_row.
    try:
        base_params = build_judge_params(session=session)
    except Exception as exc:
        logger.error(
            f"[_judge_rows] {log_prefix} Judge setup failed; leaving all rows "
            f"unjudged | error={exc}",
            exc_info=True,
        )
        return results, {ref for _item_id, ref, _r in judgeable}, None

    judge_model = base_params.get("model")

    max_workers = max(1, min(settings.EVAL_FAST_API_CONCURRENCY, len(judgeable)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                judge_row,
                openai_client=openai_client,
                base_params=base_params,
                metrics=metrics,
                inputs={
                    JudgeInputEnum.CONFIG_PROMPT: config_prompt,
                    JudgeInputEnum.QUESTION: response.get("question", ""),
                    JudgeInputEnum.GENERATED_ANSWER: response.get(
                        "generated_output", ""
                    ),
                    JudgeInputEnum.GOLDEN_ANSWER: response.get("ground_truth", ""),
                    JudgeInputEnum.RETRIEVED_CHUNKS: "\n---\n".join(
                        c.get("text", "")
                        for c in (response.get("retrieved_chunks") or [])
                        if c.get("text")
                    ),
                },
            ): (item_id, ref)
            for item_id, ref, response in judgeable
        }
        for future in as_completed(future_map):
            item_id, ref = future_map[future]
            try:
                results[item_id] = future.result()
            except Exception as exc:
                failed_refs.add(ref)
                logger.warning(
                    f"[_judge_rows] {log_prefix} Judge failed for row; flagged "
                    f"unscoreable | item_id={item_id} | ref={ref} | error={exc}"
                )

    return results, failed_refs, judge_model


def _attach_metric_scores(
    *,
    spec: JudgeMetricSpec,
    judge_results: dict[str, JudgeResult],
    summary_scores: list[dict[str, Any]],
) -> None:
    """Append one metric's run-level summary score from the combined results.

    Per-row scores and reasoning are stored per trace in the trace-build loop, which
    is what the read path serves; only the aggregate belongs on the run.
    """
    values: list[float] = [
        metric_score.score
        for result in judge_results.values()
        if (metric_score := result.metrics.get(spec.key)) is not None
    ]

    if values:
        arr = np.array(values)
        summary_scores.append(
            {
                "name": spec.score_name,
                "avg": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "total_pairs": len(values),
                "data_type": "NUMERIC",
            }
        )


def _stage3_score_and_trace(
    *,
    session: Session,
    openai_client: OpenAI,
    eval_run: EvaluationRun,
    langfuse: Langfuse | None,
    response_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]] | None,
    log_prefix: str,
) -> tuple[EvaluationRun, EvaluationScore, list[dict[str, Any]]]:
    """Stage 3 — cosine (v1) or judge (v2), create traces, attach costs. Idempotent.

    Returns the run, the full score unit (summary_scores + per-trace records in the
    batch path's shape), and the Langfuse `write_items` (empty for v2). Everything is
    keyed by `ref` (trace_id when traced, else item_id) so it works without Langfuse.

    The two scoring paths are mutually exclusive, gated on `eval_run.is_judge_run`:
      - v1: cosine over the embedded pairs, per_item_scores, the Cosine summary score
        and Langfuse sync — unchanged; v1 never judges, so a judge failure can never
        block a cosine score.
      - v2: no embeddings ran, so cosine is skipped entirely (`embedding_results` is
        None/empty); one combined judge call scores every enabled metric per row,
        per_item_scores stays NULL, and nothing is written to Langfuse.
    """
    is_judge_run = eval_run.is_judge_run
    logger.info(
        f"[_stage3_score_and_trace] {log_prefix} Scoring stage 3 | "
        f"judge_run={is_judge_run}"
    )

    item_id_to_pair = {
        r["item_id"]: r for r in (embedding_results or []) if not r.get("failed")
    }

    model = resolve_model_from_config(session=session, eval_run=eval_run)
    trace_id_mapping = create_langfuse_dataset_run(
        langfuse=langfuse,
        dataset_name=eval_run.dataset_name,
        run_name=eval_run.run_name,
        results=response_results,
        model=model,
    )

    # Scoring accumulators keyed by ref (trace_id when traced, else item_id) so they
    # persist with tracing off. Unscoreable rows stay out of avg/std/total_pairs. The
    # cosine-only fields (per_item_scores, similarities, write_items) stay empty for v2.
    per_item_scores: list[dict[str, Any]] = []
    item_id_to_score: dict[str, float] = {}
    item_id_to_ref: dict[str, str] = {}
    similarities: list[float] = []
    unscoreable: dict[str, str] = {}  # {ref: reason}
    write_items: list[dict[str, Any]] = []
    summary_scores: list[dict[str, Any]] = []

    if is_judge_run:
        # v2: no cosine. A row is judgeable only with a non-empty generated AND
        # golden answer; empty sides are unscoreable and skip the judge below.
        for response in response_results:
            item_id = response["item_id"]
            ref = trace_id_mapping.get(item_id) or item_id
            item_id_to_ref[item_id] = ref
            if not response.get("generated_output"):
                unscoreable[ref] = UNSCOREABLE_EMPTY_OUTPUT
            elif not response.get("ground_truth"):
                unscoreable[ref] = UNSCOREABLE_EMPTY_GROUND_TRUTH
    else:
        for response in response_results:
            item_id = response["item_id"]
            ref = trace_id_mapping.get(item_id) or item_id
            item_id_to_ref[item_id] = ref
            embedding_pair = item_id_to_pair.get(item_id)
            has_embeddings = (
                embedding_pair is not None
                and embedding_pair.get("output_embedding") is not None
                and embedding_pair.get("ground_truth_embedding") is not None
            )
            if not has_embeddings:
                # Classify why this item cannot be scored, for the UI flag.
                if not response.get("generated_output"):
                    unscoreable[ref] = UNSCOREABLE_EMPTY_OUTPUT
                elif not response.get("ground_truth"):
                    unscoreable[ref] = UNSCOREABLE_EMPTY_GROUND_TRUTH
                else:
                    unscoreable[ref] = UNSCOREABLE_EMBEDDING_FAILED
                continue
            cosine = calculate_cosine_similarity(
                embedding_pair["output_embedding"],
                embedding_pair["ground_truth_embedding"],
            )
            similarities.append(cosine)
            item_id_to_score[item_id] = cosine
            per_item_scores.append(
                {"trace_id": trace_id_mapping.get(item_id), "cosine_similarity": cosine}
            )

        # Langfuse write list, filtered to real trace_ids (empty when untraced).
        unscoreable_writes = [
            {
                "trace_id": trace_id_mapping[item_id],
                "unscoreable": True,
                "reason": reason,
            }
            for item_id, ref in item_id_to_ref.items()
            if item_id in trace_id_mapping
            and (reason := unscoreable.get(ref)) is not None
        ]
        scored_writes = [w for w in per_item_scores if w["trace_id"] is not None]
        write_items = scored_writes + unscoreable_writes

        # Durable source of truth, keyed by ref, persisted by the commit below.
        eval_run.per_item_scores = {
            item_id_to_ref[item_id]: round(float(score), 6)
            for item_id, score in item_id_to_score.items()
        }

        # Aggregate similarity stats, in the batch path's summary_scores shape.
        if similarities:
            sim_array = np.array(similarities)
            avg = float(np.mean(sim_array))
            std = float(np.std(sim_array))
        else:
            avg = 0.0
            std = 0.0

        summary_scores = apply_cosine_breakdown(
            [
                {
                    "name": COSINE_SCORE_NAME,
                    "avg": round(avg, 2),
                    "std": round(std, 2),
                    "total_pairs": len(similarities),
                    "data_type": "NUMERIC",
                }
            ],
            total_items=eval_run.total_items,
            unscoreable=unscoreable or None,
        )

    # Attach response- and embedding-stage costs (attach_cost is idempotent per stage).
    if response_results:
        attach_cost(
            session=session,
            eval_run=eval_run,
            log_prefix=log_prefix,
            response_model=model,
            response_results=response_results,
        )

    # attach_cost expects the raw OpenAI batch shape; rebuild it from embedding_results.
    if embedding_results:
        embedding_raw = [
            {
                "response": {
                    "body": {
                        "usage": r.get("usage")
                        or {"prompt_tokens": 0, "total_tokens": 0}
                    }
                }
            }
            for r in embedding_results
            if not r.get("failed")
        ]
        if embedding_raw:
            attach_cost(
                session=session,
                eval_run=eval_run,
                log_prefix=log_prefix,
                embedding_model=EMBEDDING_MODEL,
                embedding_raw_results=embedding_raw,
            )

    judge_results: dict[str, JudgeResult] = {}
    # Stays empty for v1, which never judges.
    metrics: list[JudgeMetricSpec] = []
    if eval_run.is_judge_run:
        judgeable = [
            (response["item_id"], item_id_to_ref[response["item_id"]], response)
            for response in response_results
            if response.get("generated_output") and response.get("ground_truth")
        ]

        # Run-level input: resolved once for every row. When missing, only the
        # metrics requiring it drop out; the run still completes.
        config_prompt = _resolve_config_prompt(
            session=session, eval_run=eval_run, log_prefix=log_prefix
        )
        available_run_inputs = (
            frozenset({JudgeInputEnum.CONFIG_PROMPT}) if config_prompt else frozenset()
        )
        metrics = enabled_metric_specs(available_run_inputs=available_run_inputs)

        judge_results, judge_failed_refs, judge_model = _judge_rows(
            session=session,
            openai_client=openai_client,
            metrics=metrics,
            config_prompt=config_prompt or "",
            judgeable=judgeable,
            log_prefix=log_prefix,
        )

        # Flag judge-failed rows unscoreable WITHOUT clobbering an empty-side reason
        # (setdefault): a row already flagged empty_output/empty_ground_truth keeps it.
        for ref in judge_failed_refs:
            unscoreable.setdefault(ref, JUDGE_FAILED_REASON)

        for spec in metrics:
            _attach_metric_scores(
                spec=spec,
                judge_results=judge_results,
                summary_scores=summary_scores,
            )

        # One combined call grades every metric, so its tokens can't be split per
        # metric; see JUDGE_COST_STAGE.
        if judge_results and judge_model:
            attach_cost(
                session=session,
                eval_run=eval_run,
                log_prefix=log_prefix,
                judge_stage=JUDGE_COST_STAGE,
                judge_model=judge_model,
                judge_results=[
                    {"usage": result.usage} for result in judge_results.values()
                ],
            )

    eval_run.unscoreable = unscoreable or None

    # Per-trace records, in the batch path's shape. Keyed by ref so untraced runs
    # persist too. Judge metric scores carry their reasoning in the score comment.
    traces: list[TraceData] = []
    for response in response_results:
        item_id = response["item_id"]
        ref = item_id_to_ref.get(item_id, item_id)
        trace_scores: list[TraceScore] = []
        # v2 carries no cosine score or placeholder — only the judge scores below.
        if not is_judge_run:
            cosine = item_id_to_score.get(item_id)
            if cosine is not None:
                trace_scores.append(
                    {
                        "name": COSINE_SCORE_NAME,
                        "value": round(cosine, 2),
                        "data_type": "NUMERIC",
                        "comment": COSINE_SCORE_COMMENT,
                    }
                )
            elif ref in unscoreable and unscoreable[ref] != JUDGE_FAILED_REASON:
                # Placeholder 0-score, excluded from summary stats via the marker. A
                # judge_failed-only reason is about the judge, not cosine, so it gets
                # no cosine placeholder.
                trace_scores.append(
                    {
                        "name": COSINE_SCORE_NAME,
                        "value": 0,
                        "data_type": "NUMERIC",
                        "comment": f"Cannot compute: {unscoreable[ref]}",
                        "unscoreable": True,
                    }
                )

        judge_result = judge_results.get(item_id)
        if judge_result is not None:
            sorted_chunks = sorted(
                response.get("retrieved_chunks") or [],
                key=lambda c: c.get("score", 0),
                reverse=True,
            )
            top_matches = _format_top_kb_matches(sorted_chunks)
            for spec in metrics:
                metric_score = judge_result.metrics.get(spec.key)
                is_kb = spec.key == JudgeMetricEnum.KNOWLEDGE_BASE
                if metric_score is not None:
                    comment = metric_score.reasoning
                    if is_kb:
                        comment = f"{comment} | Top matches: {top_matches}"
                    trace_scores.append(
                        {
                            "name": spec.score_name,
                            "value": round(metric_score.score, 2),
                            "data_type": "NUMERIC",
                            "comment": comment,
                        }
                    )
                elif is_kb:
                    # KB dropped for this row: surface a human reason instead of a bare
                    # N/A. Placeholder lives only in trace_scores, never the summary avg.
                    if not sorted_chunks:
                        # ponytail: empty chunks under auto tool_choice ~= not queried;
                        # a "was queried" flag would disambiguate an empty-store hit, not
                        # worth plumbing.
                        reason = "Knowledge base not queried."
                    else:
                        # Chunks present but the judge returned no KB score (rare: a
                        # well-formed reply that omitted the metric).
                        reason = "Knowledge base score unavailable for this row."
                    trace_scores.append(
                        {
                            "name": spec.score_name,
                            "value": "N/A",
                            "data_type": "CATEGORICAL",
                            "comment": reason,
                            "unscoreable": True,
                        }
                    )

        traces.append(
            {
                "trace_id": ref,
                "question": response.get("question", ""),
                "llm_answer": response.get("generated_output", ""),
                "ground_truth_answer": response.get("ground_truth", ""),
                "question_id": response.get("question_id"),
                "scores": trace_scores,
            }
        )

    # Persist cost + unscoreable here; the score unit (summary + traces) is persisted
    # by the caller via save_score so it lands in S3 like the batch path.
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(
            cost=eval_run.cost,
            unscoreable=eval_run.unscoreable,
        ),
    )

    score: EvaluationScore = {
        "summary_scores": summary_scores,
        "traces": traces,
    }
    return eval_run, score, write_items


def run_fast_evaluation(
    *,
    session: Session,
    openai_client: OpenAI,
    langfuse: Langfuse | None,
    eval_run: EvaluationRun,
) -> EvaluationRun:
    """Merge the response chunks, then run embeddings + scoring + completion.

    Called from `run_evaluation_fast_aggregate` (the cron barrier enqueues it
    only after every chunk has a raw_output_url). Stages are skipped on retry
    when their batch_job marker is set. Raises on terminal failure (run marked
    failed).

    `langfuse` is None for v2 judged runs (fully Kaapi-native, no trace creation
    or score sync) and for tracing-opted-out projects; scoring falls back to
    keying by item_id. Whether the run judges is read from `eval_run.is_judge_run`.
    """
    log_prefix = (
        f"[org={eval_run.organization_id}]"
        f"[project={eval_run.project_id}]"
        f"[eval={eval_run.id}]"
    )
    logger.info(f"[run_fast_evaluation] {log_prefix} Starting fast eval aggregation")

    if eval_run.status == "pending":
        eval_run = update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(status="processing"),
        )

    # Stage 1 — merge the response chunks.
    eval_run, response_results = _merge_response_chunks(
        session=session,
        eval_run=eval_run,
    )

    # Failure threshold is decided over the full merged set, not per chunk.
    failed_count = sum(1 for r in response_results if r.get("failed"))
    if _is_failure_threshold_breached(
        failed_rows=failed_count, total_rows=len(response_results)
    ):
        raise RuntimeError(
            f"Fast eval exceeded failure threshold | "
            f"failed={failed_count}/{len(response_results)} | "
            f"threshold={settings.EVAL_FAST_FAILURE_THRESHOLD}"
        )

    # Stage 2 — embeddings feed only cosine, so v2 judged runs skip it entirely
    # (no embedding API calls, no embedding_batch_job). v1 embeds exactly as before.
    embedding_results: list[dict[str, Any]] | None = None
    if not eval_run.is_judge_run:
        eval_run, embedding_results = _stage2_embeddings(
            session=session,
            openai_client=openai_client,
            eval_run=eval_run,
            response_results=response_results,
            log_prefix=log_prefix,
        )

    # Stage 3
    eval_run, score, write_items = _stage3_score_and_trace(
        session=session,
        openai_client=openai_client,
        eval_run=eval_run,
        langfuse=langfuse,
        response_results=response_results,
        embedding_results=embedding_results,
        log_prefix=log_prefix,
    )

    # Stage 4 — mark completed WITH the summary score so there's never a
    # completed + NULL-score window. Cost was persisted in Stage 3.
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(
            status="completed",
            score={"summary_scores": score["summary_scores"]},
            cost=eval_run.cost,
        ),
    )

    _cleanup_response_chunks(session=session, eval_run=eval_run)

    # Stage 5a — write cosine scores to Langfuse after completion (mirrors the
    # batch path). is_score_updated tracks the outcome so a cron can retry the
    # gap from per_item_scores. Skipped entirely when langfuse is None — v2 judged
    # runs are Kaapi-native and never sync scores to Langfuse.
    is_score_updated = True
    if langfuse is not None and write_items:
        try:
            failed_trace_ids = update_traces_with_cosine_scores(
                langfuse=langfuse, per_item_scores=write_items
            )
            if failed_trace_ids:
                is_score_updated = False
                logger.warning(
                    f"[run_fast_evaluation] {log_prefix} "
                    f"{len(failed_trace_ids)} Langfuse score writes failed; "
                    f"recoverable from durable per_item_scores on resync"
                )
        except Exception as exc:
            # Score-update failures don't fail the run (score lives in eval_run.score).
            is_score_updated = False
            logger.warning(
                f"[run_fast_evaluation] {log_prefix} "
                f"Failed to update Langfuse traces with scores | error={exc}",
                exc_info=True,
            )
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(is_score_updated=is_score_updated),
    )

    # Stage 5b — persist the score unit (traces to S3, summary to DB) so the read
    # path serves the cached unit instead of racing Langfuse ingestion.
    saved = save_score(
        eval_run_id=eval_run.id,
        organization_id=eval_run.organization_id,
        project_id=eval_run.project_id,
        score=score,
    )
    if saved is not None:
        eval_run = saved
        eval_run.score = score

    logger.info(
        f"[run_fast_evaluation] {log_prefix} Fast evaluation completed | "
        f"total_items={eval_run.total_items}"
    )
    return eval_run
