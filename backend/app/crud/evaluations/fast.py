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

This module owns orchestration and IO. The per-item shapes live in
`fast_results`, the `batch_job` bookkeeping in `fast_chunks`, trace records in
`fast_traces`, and the two mutually exclusive scoring paths in `fast_cosine`
(v1) and `judge_stage` (v2).

See `Fast Evaluation SRD.md` for the full design.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, cast

import openai
from langfuse import Langfuse
from openai import OpenAI
from sqlmodel import Session

from app.core.cloud.storage import get_cloud_storage
from app.core.config import settings
from app.core.storage_utils import (
    load_json_from_object_store,
    upload_jsonl_to_object_store,
)
from app.crud.evaluations.core import (
    resolve_model_from_config,
    save_score,
    update_evaluation_run,
)
from app.crud.evaluations.cost import attach_cost
from app.crud.evaluations.dataset import (
    DATASET_META_DUPLICATION_FACTOR,
    get_dataset_by_id,
)
from app.crud.evaluations.embeddings import EMBEDDING_MODEL
from app.crud.evaluations.fast_chunks import (  # noqa: F401  (re-exported)
    CHUNK_CONFIG_INDEX,
    CHUNK_CONFIG_RUN_ID,
    JOB_TYPE_EMBEDDING_FAST,
    JOB_TYPE_EVALUATION_FAST,
    JOB_TYPE_EVALUATION_FAST_CHUNK,
    create_embedding_job,
    create_merged_response_job,
    create_response_chunk_job,
    delete_response_chunk_artifacts,
    get_chunk_job,
    list_response_chunk_jobs,
)
from app.crud.evaluations.fast_cosine import (
    build_item_refs,
    classify_empty_side,
    score_cosine_run,
)
from app.crud.evaluations.fast_results import (
    EMBEDDING_USAGE_KEYS,
    RESPONSE_USAGE_KEYS,
    build_embedding_failure,
    build_response_result,
    extract_usage,
    is_failure_threshold_breached,
    parse_embedding_pair,
)
from app.crud.evaluations.fast_traces import build_trace_records, format_top_kb_matches
from app.crud.evaluations.judge import METRIC_REGISTRY, JudgeMetricSpec, JudgeResult
from app.crud.evaluations.judge_stage import (  # noqa: F401  (re-exported)
    PROMPT_TEMPLATE_LABEL,
    build_metric_summary_scores,
    judge_rows,
    resolve_config_prompt,
    select_judgeable_rows,
)
from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
)
from app.crud.evaluations.response_parsing import extract_response_text
from app.crud.evaluations.retry import retry_openai_call
from app.crud.evaluations.score import (
    JUDGE_FAILED_REASON,
    EvaluationScore,
    OverallSummary,
    SummaryScore,
    TraceData,
    compute_overall_summary,
)
from app.crud.evaluations.summary import generate_run_ai_summary
from app.crud.job import get_batch_job
from app.models import EvaluationRun, EvaluationRunUpdate
from app.models.batch_job import BatchJob
from app.models.llm.request import TextLLMParams
from app.services.llm.mappers import map_kaapi_to_openai_params
from app.services.response.response import get_file_search_results

logger = logging.getLogger(__name__)

# The job-type/chunk-config constants above and these aliases are re-exported:
# callers and tests still import them from this module, not from the split-out ones.
_format_top_kb_matches = format_top_kb_matches
_get_chunk_job = get_chunk_job
_is_failure_threshold_breached = is_failure_threshold_breached

_retry_openai_call = retry_openai_call(logger)


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


def _run_in_pool(
    *,
    items: list[Any],
    worker: Callable[[Any], dict[str, Any]],
    max_workers: int,
) -> list[dict[str, Any]]:
    """Fan `worker` out over `items` in a thread pool, collecting every result."""
    results: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(items) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _log_prefix(eval_run: EvaluationRun) -> str:
    return (
        f"[org={eval_run.organization_id}]"
        f"[project={eval_run.project_id}]"
        f"[eval={eval_run.id}]"
    )


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

    def failed_result(generated_output: str) -> dict[str, Any]:
        return build_response_result(
            item_id=item_id,
            question=question,
            ground_truth=ground_truth,
            question_id=question_id,
            generated_output=generated_output,
            failed=True,
        )

    if not question:
        return failed_result("ERROR: missing question in dataset item")

    try:
        response = _create_response(openai_client, {**base_params, "input": question})
    except openai.OpenAIError as exc:
        logger.warning(
            f"[_responses_call_for_item] Item failed | item_id={item_id} | error={exc}"
        )
        return failed_result(f"ERROR: {exc}")

    return build_response_result(
        item_id=item_id,
        question=question,
        ground_truth=ground_truth,
        question_id=question_id,
        generated_output=extract_response_text(response),
        response_id=getattr(response, "id", None),
        usage=extract_usage(getattr(response, "usage", None), RESPONSE_USAGE_KEYS),
        failed=False,
        # Plain dicts (not FileResultChunk) so the unit stays JSON-serializable for S3.
        retrieved_chunks=[
            {"score": c.score, "text": c.text, "filename": c.filename}
            for c in get_file_search_results(response)
        ],
    )


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
        return build_embedding_failure(item_id, "empty output or ground_truth")

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
        return build_embedding_failure(item_id, str(exc))

    return parse_embedding_pair(item_id=item_id, response=response)


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


def _cleanup_response_chunks(*, session: Session, eval_run: EvaluationRun) -> None:
    """Drop the per-chunk S3 files + batch_job rows once a run completes.

    Best-effort end to end: this runs after the completed transition but before
    save_score, so anything raising here would both flip a completed run to
    failed and cost it its score unit. Resolving storage is guarded for the same
    reason the deletes are.
    """
    try:
        storage = get_cloud_storage(session=session, project_id=eval_run.project_id)
    except Exception as exc:
        logger.warning(
            f"[_cleanup_response_chunks] Cleanup skipped (orphans harmless) | "
            f"eval_run_id={eval_run.id} | error={exc}",
            exc_info=True,
        )
        return

    delete_response_chunk_artifacts(
        session=session, storage=storage, eval_run_id=eval_run.id
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
        session=session, kaapi_params=config
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

    results = _run_in_pool(
        items=dataset_items_slice,
        worker=lambda item: _responses_call_for_item(
            openai_client=openai_client, base_params=base_params, item=item
        ),
        max_workers=settings.EVAL_FAST_API_CONCURRENCY,
    )

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
    create_response_chunk_job(
        session=session,
        eval_run=eval_run,
        chunk_index=chunk_index,
        model=config.model,
        results=results,
        raw_output_url=raw_output_url,
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
    log_prefix = _log_prefix(eval_run)
    cached = _load_completed_stage(
        session=session,
        batch_job_id=eval_run.batch_job_id,
        project_id=eval_run.project_id,
        log_prefix=log_prefix,
        stage="_merge_response_chunks",
    )
    if cached is not None:
        return eval_run, cached

    chunk_job_by_index: dict[int, BatchJob] = {}
    for job in list_response_chunk_jobs(session=session, eval_run_id=eval_run.id):
        chunk_index = int(job.config.get(CHUNK_CONFIG_INDEX, -1))
        if job.raw_output_url and chunk_index not in chunk_job_by_index:
            chunk_job_by_index[chunk_index] = job

    results: list[dict[str, Any]] = []
    for chunk_index in sorted(chunk_job_by_index):
        raw_output_url = chunk_job_by_index[chunk_index].raw_output_url
        assert raw_output_url is not None  # guaranteed by the filter above
        results.extend(
            _load_unit_from_s3(
                session=session, project_id=eval_run.project_id, url=raw_output_url
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
    batch_job = create_merged_response_job(
        session=session,
        eval_run=eval_run,
        model=model,
        results=results,
        raw_output_url=raw_output_url,
    )

    # batch_job_id / total_items aren't on EvaluationRunUpdate; set them directly.
    eval_run.batch_job_id = batch_job.id
    eval_run.total_items = len(results)
    eval_run = update_evaluation_run(
        session=session, eval_run=eval_run, update=EvaluationRunUpdate()
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

    embedding_results = _run_in_pool(
        items=embed_candidates,
        worker=lambda result: _embedding_call_for_pair(
            openai_client=openai_client,
            embedding_model=EMBEDDING_MODEL,
            item_id=result["item_id"],
            output_text=result.get("generated_output", ""),
            ground_truth=result.get("ground_truth", ""),
        ),
        max_workers=settings.EVAL_FAST_API_CONCURRENCY,
    )

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
    batch_job = create_embedding_job(
        session=session,
        eval_run=eval_run,
        embedding_model=EMBEDDING_MODEL,
        results=embedding_results,
        raw_output_url=raw_output_url,
    )
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(embedding_batch_job_id=batch_job.id),
    )
    return eval_run, embedding_results


@dataclass
class _ScoringOutcome:
    """What one scoring path (cosine or judge) contributes to Stage 3."""

    summary_scores: list[SummaryScore] = field(default_factory=list)
    unscoreable: dict[str, str] = field(default_factory=dict)
    write_items: list[dict[str, Any]] = field(default_factory=list)
    cosine_by_item_id: dict[str, float] = field(default_factory=dict)
    judge_results: dict[str, JudgeResult] = field(default_factory=dict)
    metrics: list[JudgeMetricSpec] = field(default_factory=list)
    config_prompt: str = ""


def _attach_stage_costs(
    *,
    session: Session,
    eval_run: EvaluationRun,
    log_prefix: str,
    model: str | None,
    response_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]] | None,
) -> None:
    """Attach the response- and embedding-stage costs (idempotent per stage)."""
    if response_results:
        attach_cost(
            session=session,
            eval_run=eval_run,
            log_prefix=log_prefix,
            response_model=model,
            response_results=response_results,
        )

    # attach_cost expects the raw OpenAI batch shape; rebuild it from embedding_results.
    embedding_raw = [
        {
            "response": {
                "body": {
                    "usage": r.get("usage") or dict.fromkeys(EMBEDDING_USAGE_KEYS, 0)
                }
            }
        }
        for r in (embedding_results or [])
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


def _score_cosine_path(
    *,
    response_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]] | None,
    item_refs: dict[str, str],
    trace_id_mapping: dict[str, str],
    eval_run: EvaluationRun,
) -> _ScoringOutcome:
    """v1 — cosine over the embedded pairs, plus the Langfuse write list."""
    cosine = score_cosine_run(
        response_results=response_results,
        embedding_results=embedding_results,
        item_refs=item_refs,
        trace_id_mapping=trace_id_mapping,
        total_items=eval_run.total_items,
    )
    # Durable source of truth, keyed by ref, persisted by the Stage 3 commit.
    eval_run.per_item_scores = cosine.per_item_scores
    return _ScoringOutcome(
        summary_scores=cosine.summary_scores,
        unscoreable=cosine.unscoreable,
        write_items=cosine.write_items,
        cosine_by_item_id=cosine.item_id_to_score,
    )


def _score_judge_path(
    *,
    session: Session,
    openai_client: OpenAI,
    response_results: list[dict[str, Any]],
    item_refs: dict[str, str],
    eval_run: EvaluationRun,
    log_prefix: str,
) -> _ScoringOutcome:
    """v2 — one combined judge call per row; no cosine, no Langfuse writes."""
    outcome = _ScoringOutcome(metrics=list(METRIC_REGISTRY.values()))

    # A row is judgeable only with a non-empty generated AND golden answer.
    for response in response_results:
        reason = classify_empty_side(response)
        if reason is not None:
            outcome.unscoreable[item_refs[response["item_id"]]] = reason

    # Run-level input, resolved once for every row. When it resolves to None the
    # prompt metric drops out per row (empty input); the run still completes.
    outcome.config_prompt = (
        resolve_config_prompt(session=session, eval_run=eval_run, log_prefix=log_prefix)
        or ""
    )

    outcome.judge_results, judge_failed_refs, judge_model = judge_rows(
        session=session,
        openai_client=openai_client,
        metrics=outcome.metrics,
        config_prompt=outcome.config_prompt,
        judgeable=select_judgeable_rows(response_results),
        item_refs=item_refs,
        log_prefix=log_prefix,
    )

    # setdefault so a row already flagged empty_output/empty_ground_truth keeps it.
    for ref in judge_failed_refs:
        outcome.unscoreable.setdefault(ref, JUDGE_FAILED_REASON)

    outcome.summary_scores = build_metric_summary_scores(
        metrics=outcome.metrics, judge_results=outcome.judge_results
    )

    # One combined call grades every metric, so its tokens can't be split per
    # metric — they land in a single "judge" cost stage.
    if outcome.judge_results and judge_model:
        attach_cost(
            session=session,
            eval_run=eval_run,
            log_prefix=log_prefix,
            judge_model=judge_model,
            judge_results=[
                {"usage": result.usage} for result in outcome.judge_results.values()
            ],
        )
    return outcome


def _effective_duplication_factor(*, session: Session, eval_run: EvaluationRun) -> int:
    """The run's duplication factor, falling back to the dataset's stored one.

    Falls back to 1 (no repetition) when neither resolves, so the summary still
    generates.
    """
    if eval_run.duplication_factor is not None:
        return max(1, eval_run.duplication_factor)

    dataset = get_dataset_by_id(
        session=session,
        dataset_id=eval_run.dataset_id,
        organization_id=eval_run.organization_id,
        project_id=eval_run.project_id,
    )
    metadata = dataset.dataset_metadata if dataset else None
    return max(1, int((metadata or {}).get(DATASET_META_DUPLICATION_FACTOR, 1)))


def _build_overall_summary(
    *,
    session: Session,
    eval_run: EvaluationRun,
    outcome: _ScoringOutcome,
    traces: list[TraceData],
) -> OverallSummary | None:
    """Run-level weighted rollup, plus the best-effort AI note diagnosing the traces."""
    avg_by_name = {s["name"]: s["avg"] for s in outcome.summary_scores if "avg" in s}
    overall = compute_overall_summary(
        metric_avgs={
            spec.key.value: avg_by_name[spec.score_name]
            for spec in outcome.metrics
            if spec.score_name in avg_by_name
        },
        metric_weights={spec.key.value: spec.weight for spec in outcome.metrics},
        metric_names={spec.key.value: spec.score_name for spec in outcome.metrics},
    )
    if overall is None:
        return None

    overall["ai_summary"] = generate_run_ai_summary(
        model=settings.EVAL_SUMMARY_MODEL,
        run_name=eval_run.run_name,
        duplication_factor=_effective_duplication_factor(
            session=session, eval_run=eval_run
        ),
        config_prompt=outcome.config_prompt,
        traces=traces,
    )
    return overall


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
    batch path's shape), and the Langfuse `write_items` (empty for v2). Everything
    is keyed by `ref` (trace_id when traced, else item_id) so it works without
    Langfuse.

    The two scoring paths are mutually exclusive, gated on `eval_run.is_judge_run`:
      - v1: cosine over the embedded pairs, per_item_scores, the Cosine summary
        score and Langfuse sync — unchanged; v1 never judges, so a judge failure
        can never block a cosine score.
      - v2: no embeddings ran, so cosine is skipped entirely; one combined judge
        call scores every enabled metric per row, per_item_scores stays NULL, and
        nothing is written to Langfuse.
    """
    is_judge_run = eval_run.is_judge_run
    logger.info(
        f"[_stage3_score_and_trace] {log_prefix} Scoring stage 3 | "
        f"judge_run={is_judge_run}"
    )

    model = resolve_model_from_config(session=session, eval_run=eval_run)
    trace_id_mapping = create_langfuse_dataset_run(
        langfuse=langfuse,
        dataset_name=eval_run.dataset_name,
        run_name=eval_run.run_name,
        results=response_results,
        model=model,
    )
    item_refs = build_item_refs(response_results, trace_id_mapping)

    def attach_costs() -> None:
        _attach_stage_costs(
            session=session,
            eval_run=eval_run,
            log_prefix=log_prefix,
            model=model,
            response_results=response_results,
            embedding_results=embedding_results,
        )

    if is_judge_run:
        # Response cost lands before the judge's own cost stage.
        attach_costs()
        outcome = _score_judge_path(
            session=session,
            openai_client=openai_client,
            response_results=response_results,
            item_refs=item_refs,
            eval_run=eval_run,
            log_prefix=log_prefix,
        )
    else:
        outcome = _score_cosine_path(
            response_results=response_results,
            embedding_results=embedding_results,
            item_refs=item_refs,
            trace_id_mapping=trace_id_mapping,
            eval_run=eval_run,
        )
        attach_costs()

    eval_run.unscoreable = outcome.unscoreable or None

    traces = build_trace_records(
        response_results=response_results,
        item_refs=item_refs,
        is_judge_run=is_judge_run,
        judge_results=outcome.judge_results,
        metrics=outcome.metrics,
        cosine_by_item_id=outcome.cosine_by_item_id,
        unscoreable=outcome.unscoreable,
    )

    # Runs after the traces, since the AI summary diagnoses them.
    overall = (
        _build_overall_summary(
            session=session, eval_run=eval_run, outcome=outcome, traces=traces
        )
        if is_judge_run
        else None
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
        "summary_scores": outcome.summary_scores,
        "traces": traces,
    }
    if overall is not None:
        score["overall"] = overall
    return eval_run, score, outcome.write_items


def _sync_scores_to_langfuse(
    *, langfuse: Langfuse | None, write_items: list[dict[str, Any]], log_prefix: str
) -> bool:
    """Write cosine scores back to Langfuse; False if any write was lost.

    Never fails the run — the score already lives on `eval_run`, so a cron can
    retry the gap from the durable per_item_scores map.
    """
    if langfuse is None or not write_items:
        return True

    try:
        failed_trace_ids = update_traces_with_cosine_scores(
            langfuse=langfuse, per_item_scores=write_items
        )
    except Exception as exc:
        logger.warning(
            f"[_sync_scores_to_langfuse] {log_prefix} Failed to update Langfuse "
            f"traces with scores | error={exc}",
            exc_info=True,
        )
        return False

    if failed_trace_ids:
        logger.warning(
            f"[_sync_scores_to_langfuse] {log_prefix} {len(failed_trace_ids)} "
            f"Langfuse score writes failed; recoverable from durable "
            f"per_item_scores on resync"
        )
        return False
    return True


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
    log_prefix = _log_prefix(eval_run)
    logger.info(f"[run_fast_evaluation] {log_prefix} Starting fast eval aggregation")

    if eval_run.status == "pending":
        eval_run = update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(status="processing"),
        )

    # Stage 1 — merge the response chunks.
    eval_run, response_results = _merge_response_chunks(
        session=session, eval_run=eval_run
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
            # Persist the overall alongside the summary so GET run status shows the
            # run-level score/verdict/breakdown without loading the S3 trace unit.
            score={
                "summary_scores": score["summary_scores"],
                "overall": score.get("overall"),
            },
            cost=eval_run.cost,
        ),
    )

    _cleanup_response_chunks(session=session, eval_run=eval_run)

    # Stage 5a — write cosine scores to Langfuse after completion (mirrors the
    # batch path). is_score_updated tracks the outcome so a cron can retry the gap.
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(
            is_score_updated=_sync_scores_to_langfuse(
                langfuse=langfuse, write_items=write_items, log_prefix=log_prefix
            )
        ),
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
        eval_run.score = cast(dict[str, object], score)

    logger.info(
        f"[run_fast_evaluation] {log_prefix} Fast evaluation completed | "
        f"total_items={eval_run.total_items}"
    )
    return eval_run
