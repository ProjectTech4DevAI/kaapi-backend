"""Fast evaluation orchestration (run_mode="fast").

Synchronous text-eval path: makes Responses + Embeddings calls in parallel from
a single Celery task and persists per-stage units to S3. Each stage is skipped
on retry if its `batch_job` row already exists.

    Stage 1 — Responses unit:   evaluation_run.batch_job_id
    Stage 2 — Embeddings unit:  evaluation_run.embedding_batch_job_id
    Stage 3 — Score + trace + cost (no marker; each step is idempotent)
    Stage 4 — Mark completed

See `Fast Evaluation SRD.md` for the full design.
"""

import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import numpy as np
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
from app.crud.evaluations.batch import fetch_dataset_items
from app.crud.evaluations.core import resolve_model_from_config, update_evaluation_run
from app.crud.evaluations.cost import attach_cost
from app.crud.evaluations.embeddings import (
    EMBEDDING_MODEL,
    calculate_cosine_similarity,
)
from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
)
from app.crud.job import create_batch_job, get_batch_job
from app.models import EvaluationRun, EvaluationRunUpdate
from app.models.batch_job import BatchJobCreate
from app.models.evaluation import RunModeEnum
from app.models.llm.request import TextLLMParams

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# job_type discriminators on batch_job for the two fast-path stages. The row's
# presence + raw_output_url is what marks a stage as already done on retry.
JOB_TYPE_EVALUATION_FAST = "evaluation_fast"
JOB_TYPE_EMBEDDING_FAST = "embedding_fast"


# Per-call retry policy for Stage 1 / Stage 2.
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 30.0

_RETRYABLE_OPENAI_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _sleep_with_backoff(attempt: int) -> None:
    """Exponential backoff with full jitter, capped at _RETRY_MAX_DELAY_SECONDS."""
    base = min(
        _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), _RETRY_MAX_DELAY_SECONDS
    )
    delay = random.uniform(0, base)
    time.sleep(delay)


def _call_with_retry(label: str, fn: Callable[[], _T]) -> _T:
    """Call `fn()`, retrying transient OpenAI errors; permanent errors fail fast."""
    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except _RETRYABLE_OPENAI_ERRORS as exc:
            if attempt == _RETRY_MAX_ATTEMPTS:
                logger.warning(
                    f"[_call_with_retry] Exhausted retries | label={label} | "
                    f"attempt={attempt} | error={exc}"
                )
                raise
            logger.info(
                f"[_call_with_retry] Transient error, retrying | label={label} | "
                f"attempt={attempt} | error={exc}"
            )
            _sleep_with_backoff(attempt)
        except openai.OpenAIError as exc:
            logger.warning(
                f"[_call_with_retry] Permanent error, no retry | label={label} | "
                f"error={exc}"
            )
            raise
    # Unreachable: the loop always returns or raises. Keeps mypy happy.
    raise RuntimeError(f"_call_with_retry exited without result for {label}")


def _build_responses_params(
    *,
    config: TextLLMParams,
    question: str,
) -> dict[str, Any]:
    """Build params for one Responses call, mirroring the batch path's body shape."""
    params: dict[str, Any] = {
        "model": config.model,
        "instructions": config.instructions,
        "input": question,
    }

    if "temperature" in config.model_fields_set:
        params["temperature"] = config.temperature

    if config.reasoning:
        params["reasoning"] = {"effort": config.reasoning}

    if config.knowledge_base_ids:
        params["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": config.knowledge_base_ids,
                "max_num_results": config.max_num_results or 20,
            }
        ]

    return params


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


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Normalize an OpenAI usage object into the cost layer's dict shape."""
    return {
        "input_tokens": int(_field(usage, "input_tokens", 0) or 0),
        "output_tokens": int(_field(usage, "output_tokens", 0) or 0),
        "total_tokens": int(_field(usage, "total_tokens", 0) or 0),
    }


def _responses_call_for_item(
    *,
    openai_client: OpenAI,
    config: TextLLMParams,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Run one Responses call for a dataset item, in the batch path's per-item shape."""
    item_id = item["id"]
    question = item["input"].get("question", "") if item.get("input") else ""
    ground_truth = (
        item["expected_output"].get("answer", "") if item.get("expected_output") else ""
    )
    question_id = (item.get("metadata") or {}).get("question_id")

    if not question:
        return {
            "item_id": item_id,
            "question": "",
            "generated_output": "ERROR: missing question in dataset item",
            "ground_truth": ground_truth,
            "response_id": None,
            "usage": None,
            "question_id": question_id,
            "failed": True,
        }

    params = _build_responses_params(config=config, question=question)

    try:
        response = _call_with_retry(
            label=f"responses.create:{item_id}",
            fn=lambda: openai_client.responses.create(**params),
        )
    except openai.OpenAIError as exc:
        logger.warning(
            f"[_responses_call_for_item] Item failed | item_id={item_id} | error={exc}"
        )
        return {
            "item_id": item_id,
            "question": question,
            "generated_output": f"ERROR: {exc}",
            "ground_truth": ground_truth,
            "response_id": None,
            "usage": None,
            "question_id": question_id,
            "failed": True,
        }

    return {
        "item_id": item_id,
        "question": question,
        "generated_output": _extract_response_text(response),
        "ground_truth": ground_truth,
        "response_id": getattr(response, "id", None),
        "usage": _usage_to_dict(getattr(response, "usage", None)),
        "question_id": question_id,
        "failed": False,
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
        return {
            "item_id": item_id,
            "output_embedding": None,
            "ground_truth_embedding": None,
            "usage": None,
            "failed": True,
            "error": "empty output or ground_truth",
        }

    try:
        response = _call_with_retry(
            label=f"embeddings.create:{item_id}",
            fn=lambda: openai_client.embeddings.create(
                model=embedding_model,
                input=[output_text, ground_truth],
                encoding_format="float",
            ),
        )
    except openai.OpenAIError as exc:
        logger.warning(
            f"[_embedding_call_for_pair] Item failed | item_id={item_id} | error={exc}"
        )
        return {
            "item_id": item_id,
            "output_embedding": None,
            "ground_truth_embedding": None,
            "usage": None,
            "failed": True,
            "error": str(exc),
        }

    data = _field(response, "data") or []
    if len(data) < 2:
        return {
            "item_id": item_id,
            "output_embedding": None,
            "ground_truth_embedding": None,
            "usage": None,
            "failed": True,
            "error": f"expected 2 embeddings, got {len(data)}",
        }

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


def _stage1_responses(
    *,
    session: Session,
    openai_client: OpenAI,
    eval_run: EvaluationRun,
    config: TextLLMParams,
    dataset_items: list[dict[str, Any]],
    log_prefix: str,
) -> tuple[EvaluationRun, list[dict[str, Any]]]:
    """Stage 1 — one Responses call per dataset item.

    Skipped on retry if `eval_run.batch_job_id` is set; the unit is reloaded from S3.
    """
    if eval_run.batch_job_id:
        existing = get_batch_job(session=session, batch_job_id=eval_run.batch_job_id)
        if existing and existing.raw_output_url:
            logger.info(
                f"[_stage1_responses] {log_prefix} Skipping stage 1 (already done) | "
                f"batch_job_id={existing.id}"
            )
            results = _load_unit_from_s3(
                session=session,
                project_id=eval_run.project_id,
                url=existing.raw_output_url,
            )
            return eval_run, results

    logger.info(
        f"[_stage1_responses] {log_prefix} Running stage 1 | "
        f"items={len(dataset_items)} | model={config.model} | "
        f"concurrency={settings.EVAL_FAST_API_CONCURRENCY}"
    )

    results: list[dict[str, Any]] = []
    max_workers = max(1, min(settings.EVAL_FAST_API_CONCURRENCY, len(dataset_items)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _responses_call_for_item,
                openai_client=openai_client,
                config=config,
                item=item,
            ): item["id"]
            for item in dataset_items
        }
        for future in as_completed(futures):
            results.append(future.result())

    failed_count = sum(1 for r in results if r.get("failed"))
    logger.info(
        f"[_stage1_responses] {log_prefix} Stage 1 finished | "
        f"total={len(results)} | failed={failed_count}"
    )

    if _is_failure_threshold_breached(
        failed_rows=failed_count, total_rows=len(results)
    ):
        raise RuntimeError(
            f"Fast eval Stage 1 exceeded failure threshold | "
            f"failed={failed_count}/{len(results)} | "
            f"threshold={settings.EVAL_FAST_FAILURE_THRESHOLD}"
        )

    raw_output_url = _upload_unit_to_s3(
        session=session,
        project_id=eval_run.project_id,
        eval_run_id=eval_run.id,
        filename=f"responses_{eval_run.id}.json",
        results=results,
    )

    # Aggregate usage for the batch_job summary.
    summed_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for r in results:
        usage = r.get("usage") or {}
        for k in summed_usage:
            summed_usage[k] += int(usage.get(k, 0) or 0)

    batch_job = create_batch_job(
        session=session,
        batch_job_create=BatchJobCreate(
            provider="openai",
            job_type=JOB_TYPE_EVALUATION_FAST,
            config={
                "endpoint": "/v1/responses",
                "run_mode": RunModeEnum.FAST.value,
                "model": config.model,
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
    if eval_run.embedding_batch_job_id:
        existing = get_batch_job(
            session=session, batch_job_id=eval_run.embedding_batch_job_id
        )
        if existing and existing.raw_output_url:
            logger.info(
                f"[_stage2_embeddings] {log_prefix} Skipping stage 2 (already done) | "
                f"batch_job_id={existing.id}"
            )
            embeddings = _load_unit_from_s3(
                session=session,
                project_id=eval_run.project_id,
                url=existing.raw_output_url,
            )
            return eval_run, embeddings

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

    summed_usage = {"prompt_tokens": 0, "total_tokens": 0}
    for r in embedding_results:
        usage = r.get("usage") or {}
        for k in summed_usage:
            summed_usage[k] += int(usage.get(k, 0) or 0)

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


def _stage3_score_and_trace(
    *,
    session: Session,
    eval_run: EvaluationRun,
    langfuse: Langfuse,
    response_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    log_prefix: str,
) -> EvaluationRun:
    """Stage 3 — compute cosine, create Langfuse traces, attach costs.

    No stage marker; each step is idempotent (deterministic cosine, Langfuse
    dedupes on the observe key, attach_cost overwrites per stage).
    """
    logger.info(
        f"[_stage3_score_and_trace] {log_prefix} Computing cosine + creating traces"
    )

    item_id_to_pair = {
        r["item_id"]: r for r in embedding_results if not r.get("failed")
    }

    model = resolve_model_from_config(session=session, eval_run=eval_run)
    trace_id_mapping = create_langfuse_dataset_run(
        langfuse=langfuse,
        dataset_name=eval_run.dataset_name,
        run_name=eval_run.run_name,
        results=response_results,
        model=model,
    )

    # Per-item cosine scores keyed on Langfuse trace_id.
    per_item_scores: list[dict[str, Any]] = []
    similarities: list[float] = []
    for r in response_results:
        item_id = r["item_id"]
        pair = item_id_to_pair.get(item_id)
        trace_id = trace_id_mapping.get(item_id)
        if not pair or not trace_id:
            continue
        if (
            pair.get("output_embedding") is None
            or pair.get("ground_truth_embedding") is None
        ):
            continue
        score = calculate_cosine_similarity(
            pair["output_embedding"], pair["ground_truth_embedding"]
        )
        similarities.append(score)
        per_item_scores.append({"trace_id": trace_id, "cosine_similarity": score})

    if per_item_scores:
        try:
            update_traces_with_cosine_scores(
                langfuse=langfuse, per_item_scores=per_item_scores
            )
        except Exception as exc:
            # Score-update failures don't fail the run (score lives in eval_run.score).
            logger.warning(
                f"[_stage3_score_and_trace] {log_prefix} "
                f"Failed to update Langfuse traces with scores | error={exc}",
                exc_info=True,
            )

    # Aggregate similarity stats, in the batch path's summary_scores shape.
    if similarities:
        sim_array = np.array(similarities)
        avg = float(np.mean(sim_array))
        std = float(np.std(sim_array))
    else:
        avg = 0.0
        std = 0.0

    score_payload = {
        "summary_scores": [
            {
                "name": "Cosine Similarity",
                "avg": round(avg, 2),
                "std": round(std, 2),
                "total_pairs": len(similarities),
                "data_type": "NUMERIC",
            }
        ]
    }

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

    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(
            score=score_payload,
            cost=eval_run.cost,
        ),
    )

    return eval_run


def run_fast_evaluation(
    *,
    session: Session,
    openai_client: OpenAI,
    langfuse: Langfuse,
    eval_run: EvaluationRun,
    config: TextLLMParams,
) -> EvaluationRun:
    """Run the full fast-eval pipeline for one evaluation_run.

    Called from the `run_evaluation_fast` task. Stages are skipped on retry when
    their batch_job marker is set. Raises on terminal failure (run marked failed).
    """
    log_prefix = (
        f"[org={eval_run.organization_id}]"
        f"[project={eval_run.project_id}]"
        f"[eval={eval_run.id}]"
    )
    logger.info(f"[run_fast_evaluation] {log_prefix} Starting fast evaluation pipeline")

    if eval_run.status == "pending":
        eval_run = update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(status="processing"),
        )

    dataset_items = fetch_dataset_items(
        langfuse=langfuse, dataset_name=eval_run.dataset_name
    )
    if not dataset_items:
        raise ValueError(
            f"Dataset '{eval_run.dataset_name}' returned no items for fast eval"
        )

    # Stage 1
    eval_run, response_results = _stage1_responses(
        session=session,
        openai_client=openai_client,
        eval_run=eval_run,
        config=config,
        dataset_items=dataset_items,
        log_prefix=log_prefix,
    )

    # Stage 2
    eval_run, embedding_results = _stage2_embeddings(
        session=session,
        openai_client=openai_client,
        eval_run=eval_run,
        response_results=response_results,
        log_prefix=log_prefix,
    )

    # Stage 3
    eval_run = _stage3_score_and_trace(
        session=session,
        eval_run=eval_run,
        langfuse=langfuse,
        response_results=response_results,
        embedding_results=embedding_results,
        log_prefix=log_prefix,
    )

    # Stage 4
    eval_run = update_evaluation_run(
        session=session,
        eval_run=eval_run,
        update=EvaluationRunUpdate(status="completed"),
    )

    logger.info(
        f"[run_fast_evaluation] {log_prefix} Fast evaluation completed | "
        f"total_items={eval_run.total_items}"
    )
    return eval_run
