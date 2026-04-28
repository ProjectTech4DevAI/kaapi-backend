"""Live evaluation: chord callback that aggregates per-row results.

Runs once after every `run_eval_row` task in the chord finishes. Idempotent
on re-entry (chord callbacks have at-least-once delivery semantics) — checks
the run's terminal status first and bails if already completed/failed.
"""

import logging
from collections import Counter
from typing import Any

from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.config import settings
from app.core.db import engine
from app.core.storage_utils import upload_jsonl_to_object_store
from app.crud.evaluations.core import (
    get_evaluation_run_by_id,
    resolve_model_from_config,
    update_evaluation_run,
)
from app.crud.evaluations.cost import attach_cost
from app.crud.evaluations.embeddings import (
    EMBEDDING_MODEL,
    calculate_average_similarity,
)
from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
)
from app.models import EvaluationRunUpdate
from app.utils import get_langfuse_client

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed"}


def _summarise_failures(failures: list[dict[str, Any]]) -> str:
    counts = Counter(f.get("error") or "Unknown error" for f in failures)
    top_error, top_count = counts.most_common(1)[0]
    return f"{top_error} ({top_count}/{len(failures)} failures)"


def aggregate_results(
    *,
    eval_run_id: int,
    organization_id: int,
    project_id: int,
    row_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Finalize a live evaluation run from collected per-row results.

    Returns a small summary dict mainly for observability — Celery does not
    consume it.
    """
    log_prefix = (
        f"[aggregate_results][org={organization_id}][project={project_id}]"
        f"[eval={eval_run_id}]"
    )

    with Session(engine) as session:
        eval_run = get_evaluation_run_by_id(
            session=session,
            evaluation_id=eval_run_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if eval_run is None:
            logger.error(f"{log_prefix} EvaluationRun not found, aborting aggregation")
            return {"status": "missing", "eval_run_id": eval_run_id}

        if eval_run.status in TERMINAL_STATUSES:
            logger.info(
                f"{log_prefix} EvaluationRun already in terminal status "
                f"'{eval_run.status}', no-op (chord double-fire guard)"
            )
            return {"status": eval_run.status, "eval_run_id": eval_run_id}

        total = len(row_results)
        successes = [r for r in row_results if not r.get("error")]
        failures = [r for r in row_results if r.get("error")]

        logger.info(
            f"{log_prefix} Aggregating | total={total} | successes={len(successes)} | "
            f"failures={len(failures)}"
        )

        # Failure-threshold gate: too many bad rows -> mark whole run failed.
        if total == 0 or (
            failures and len(failures) / total > settings.EVAL_LIVE_FAILURE_THRESHOLD
        ):
            error_message = (
                _summarise_failures(failures) if failures else "No row results returned"
            )
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(
                    status="failed",
                    error_message=f"Live mode: {error_message}",
                ),
            )
            return {
                "status": "failed",
                "eval_run_id": eval_run_id,
                "error": error_message,
            }

        # Persist raw row results to object store for debugging / re-processing.
        object_store_url: str | None = None
        try:
            storage = get_cloud_storage(session=session, project_id=project_id)
            object_store_url = upload_jsonl_to_object_store(
                storage=storage,
                results=row_results,
                filename="results.jsonl",
                subdirectory=f"evaluation/live-{eval_run_id}",
            )
        except Exception as store_err:
            logger.warning(f"{log_prefix} Object store upload failed | {store_err}")

        if object_store_url:
            eval_run = update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(object_store_url=object_store_url),
            )

        # Resolve model name now (uses the same config the rows ran against).
        try:
            model = resolve_model_from_config(session=session, eval_run=eval_run)
        except Exception as e:
            logger.error(f"{log_prefix} Failed to resolve model | {e}", exc_info=True)
            update_evaluation_run(
                session=session,
                eval_run=eval_run,
                update=EvaluationRunUpdate(
                    status="failed",
                    error_message=f"Live mode: model resolution failed: {e}",
                ),
            )
            return {"status": "failed", "eval_run_id": eval_run_id, "error": str(e)}

        # Response-stage cost. Live mode -> standard ("response") pricing.
        attach_cost(
            session=session,
            eval_run=eval_run,
            log_prefix=log_prefix,
            response_model=model,
            response_results=successes,
            usage_type="response",
        )

        # Langfuse traces for successful rows. Same helper as batch path.
        langfuse = get_langfuse_client(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )
        try:
            trace_id_mapping = create_langfuse_dataset_run(
                langfuse=langfuse,
                dataset_name=eval_run.dataset_name,
                run_name=eval_run.run_name,
                results=successes,
                model=model,
            )
        except Exception as e:
            logger.error(
                f"{log_prefix} Langfuse dataset run creation failed | {e}",
                exc_info=True,
            )
            trace_id_mapping = {}

        # Inline cosine similarity from the per-row vectors the row task
        # already computed — no extra API call.
        embedding_pairs: list[dict[str, Any]] = []
        embedding_input_tokens = 0
        for r in successes:
            embedding_input_tokens += r.get("embedding_input_tokens") or 0
            output_emb = r.get("output_embedding")
            ground_truth_emb = r.get("ground_truth_embedding")
            trace_id = trace_id_mapping.get(r["item_id"])
            if not trace_id or not output_emb or not ground_truth_emb:
                continue
            embedding_pairs.append(
                {
                    "trace_id": trace_id,
                    "output_embedding": output_emb,
                    "ground_truth_embedding": ground_truth_emb,
                }
            )

        score_payload: dict[str, Any] | None = None
        if embedding_pairs:
            similarity_stats = calculate_average_similarity(
                embedding_pairs=embedding_pairs
            )
            score_payload = {
                "summary_scores": [
                    {
                        "name": "Cosine Similarity",
                        "avg": round(
                            float(similarity_stats["cosine_similarity_avg"]), 2
                        ),
                        "std": round(
                            float(similarity_stats["cosine_similarity_std"]), 2
                        ),
                        "total_pairs": similarity_stats["total_pairs"],
                        "data_type": "NUMERIC",
                    }
                ]
            }

            per_item_scores = similarity_stats.get("per_item_scores", [])
            if per_item_scores:
                try:
                    update_traces_with_cosine_scores(
                        langfuse=langfuse,
                        per_item_scores=per_item_scores,
                    )
                except Exception as e:
                    logger.error(
                        f"{log_prefix} Failed to push cosine scores to Langfuse | {e}",
                        exc_info=True,
                    )

        # Embedding-stage cost from aggregated tokens. Standard pricing.
        if embedding_input_tokens > 0:
            attach_cost(
                session=session,
                eval_run=eval_run,
                log_prefix=log_prefix,
                embedding_model=EMBEDDING_MODEL,
                embedding_input_tokens=embedding_input_tokens,
                usage_type="response",
            )

        update_payload: dict[str, Any] = {
            "status": "completed",
            "total_items": len(successes),
            "cost": eval_run.cost,
        }
        if score_payload is not None:
            update_payload["score"] = score_payload
        if failures:
            update_payload["error_message"] = (
                f"Live mode: {len(failures)}/{total} rows failed "
                f"(top error: {_summarise_failures(failures)})"
            )

        update_evaluation_run(
            session=session,
            eval_run=eval_run,
            update=EvaluationRunUpdate(**update_payload),
        )

        logger.info(
            f"{log_prefix} Completed live evaluation | successes={len(successes)} | "
            f"failures={len(failures)}"
        )

        return {
            "status": "completed",
            "eval_run_id": eval_run_id,
            "successes": len(successes),
            "failures": len(failures),
        }
