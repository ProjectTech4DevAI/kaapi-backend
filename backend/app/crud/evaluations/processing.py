"""
Evaluation batch processing orchestrator.

This module coordinates the evaluation-specific workflow:
1. Monitoring batch_job status for evaluations
2. Parsing evaluation results from batch output
3. Creating Langfuse dataset runs with traces
4. Updating evaluation_run with final status and scores
"""

import ast
import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import HTTPException
from langfuse import Langfuse
from openai import OpenAI
from sqlmodel import Session, select

from app.core.batch import (
    OpenAIBatchProvider,
    download_batch_results,
    poll_batch_status,
    upload_batch_results_to_object_store,
)
from app.crud.evaluations.batch import fetch_dataset_items
from app.crud.evaluations.core import update_evaluation_run
from app.crud.evaluations.embeddings import (
    calculate_average_similarity,
    parse_embedding_results,
    start_embedding_batch,
)
from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
)
from app.crud.job import get_batch_job
from app.models import EvaluationRun
from app.utils import get_langfuse_client, get_openai_client

logger = logging.getLogger(__name__)


def parse_evaluation_output(
    raw_results: list[dict[str, Any]], dataset_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Parse batch output into evaluation results.

    This function extracts the generated output from the batch results
    and matches it with the ground truth from the dataset.

    Args:
        raw_results: Raw results from batch provider (list of JSONL lines)
        dataset_items: Original dataset items (for matching ground truth)

    Returns:
        List of results in format:
        [
            {
                "item_id": "item_123",
                "question": "What is 2+2?",
                "generated_output": "4",
                "ground_truth": "4",
                "response_id": "resp_0b99aadfead1fb62006908e7f540c48197bd110183a347c1d8",
                "usage": {
                    "input_tokens": 69,
                    "output_tokens": 258,
                    "total_tokens": 327
                }
            },
            ...
        ]
    """
    # Create lookup map for dataset items by ID
    dataset_map = {item["id"]: item for item in dataset_items}

    results = []

    for line_num, response in enumerate(raw_results, 1):
        try:
            # Extract custom_id (which is our dataset item ID)
            item_id = response.get("custom_id")
            if not item_id:
                logger.warning(
                    f"[parse_evaluation_output] No custom_id found, skipping | line={line_num}"
                )
                continue

            # Get original dataset item
            dataset_item = dataset_map.get(item_id)
            if not dataset_item:
                logger.warning(
                    f"[parse_evaluation_output] No dataset item found | line={line_num} | item_id={item_id}"
                )
                continue

            # Extract the response body
            response_body = response.get("response", {}).get("body", {})

            # Extract response ID from response.body.id
            response_id = response_body.get("id")

            # Extract usage information for cost tracking
            usage = response_body.get("usage")

            # Handle errors in batch processing
            if response.get("error"):
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(
                    f"[parse_evaluation_output] Item had error | item_id={item_id} | {error_msg}"
                )
                generated_output = f"ERROR: {error_msg}"
            else:
                # Extract text from output (can be string, list, or complex structure)
                output = response_body.get("output", "")

                # If string, try to parse it (may be JSON or Python repr of list)
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            output = ast.literal_eval(output)
                        except (ValueError, SyntaxError):
                            # Keep as string if parsing fails
                            generated_output = output
                            output = None

                # If we have a list structure, extract text from message items
                if isinstance(output, list):
                    generated_output = ""
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            for content in item.get("content", []):
                                if (
                                    isinstance(content, dict)
                                    and content.get("type") == "output_text"
                                ):
                                    generated_output = content.get("text", "")
                                    break
                            if generated_output:
                                break
                elif output is not None:
                    # output was not a string and not a list
                    generated_output = ""
                    logger.warning(
                        f"[parse_evaluation_output] Unexpected output type | item_id={item_id} | type={type(output)}"
                    )

            # Extract question and ground truth from dataset item
            question = dataset_item["input"].get("question", "")
            ground_truth = dataset_item["expected_output"].get("answer", "")

            results.append(
                {
                    "item_id": item_id,
                    "question": question,
                    "generated_output": generated_output,
                    "ground_truth": ground_truth,
                    "response_id": response_id,
                    "usage": usage,
                }
            )

        except Exception as e:
            logger.error(
                f"[parse_evaluation_output] Unexpected error | line={line_num} | {e}"
            )
            continue

    logger.info(
        f"[parse_evaluation_output] Parsed evaluation results | results={len(results)} | output_lines={len(raw_results)}"
    )
    return results


async def process_completed_evaluation(
    eval_run: EvaluationRun,
    session: Session,
    openai_client: OpenAI,
    langfuse: Langfuse,
) -> EvaluationRun:
    """
    Process a completed evaluation batch.

    This function:
    1. Downloads batch output from provider
    2. Parses results into question/output/ground_truth format
    3. Creates Langfuse dataset run with traces
    4. Starts embedding batch for similarity scoring (keeps status as "processing")

    Args:
        eval_run: EvaluationRun database object
        session: Database session
        openai_client: Configured OpenAI client
        langfuse: Configured Langfuse client

    Returns:
        Updated EvaluationRun object (with embedding_batch_job_id set)

    Raises:
        Exception: If processing fails
    """
    log_prefix = f"[org={eval_run.organization_id}][project={eval_run.project_id}][eval={eval_run.id}]"
    logger.info(
        f"[process_completed_evaluation] {log_prefix} Processing completed evaluation"
    )

    try:
        # Step 1: Get batch_job
        if not eval_run.batch_job_id:
            raise ValueError(f"EvaluationRun {eval_run.id} has no batch_job_id")

        batch_job = get_batch_job(session=session, batch_job_id=eval_run.batch_job_id)
        if not batch_job:
            raise ValueError(
                f"BatchJob {eval_run.batch_job_id} not found for evaluation {eval_run.id}"
            )

        # Step 2: Create provider and download results
        logger.info(
            f"[process_completed_evaluation] {log_prefix} Downloading batch results | batch_job_id={batch_job.id}"
        )
        provider = OpenAIBatchProvider(client=openai_client)
        raw_results = download_batch_results(provider=provider, batch_job=batch_job)

        # Step 2a: Upload raw results to object store for evaluation_run
        object_store_url = None
        try:
            object_store_url = upload_batch_results_to_object_store(
                session=session, batch_job=batch_job, results=raw_results
            )
        except Exception as store_error:
            logger.warning(
                f"[process_completed_evaluation] {log_prefix} Object store upload failed | {store_error}"
            )

        # Step 3: Fetch dataset items (needed for matching ground truth)
        logger.info(
            f"[process_completed_evaluation] {log_prefix} Fetching dataset items | dataset={eval_run.dataset_name}"
        )
        dataset_items = fetch_dataset_items(
            langfuse=langfuse, dataset_name=eval_run.dataset_name
        )

        # Step 4: Parse evaluation results
        results = parse_evaluation_output(
            raw_results=raw_results, dataset_items=dataset_items
        )

        if not results:
            raise ValueError("No valid results found in batch output")

        # Extract model from config for cost tracking
        model = eval_run.config.get("model") if eval_run.config else None

        # Step 5: Create Langfuse dataset run with traces
        trace_id_mapping = create_langfuse_dataset_run(
            langfuse=langfuse,
            dataset_name=eval_run.dataset_name,
            run_name=eval_run.run_name,
            results=results,
            model=model,
        )

        # Store object store URL in database
        if object_store_url:
            eval_run.object_store_url = object_store_url
            session.add(eval_run)
            session.commit()

        # Step 6: Start embedding batch for similarity scoring
        # Pass trace_id_mapping directly without storing in DB
        try:
            eval_run = start_embedding_batch(
                session=session,
                openai_client=openai_client,
                eval_run=eval_run,
                results=results,
                trace_id_mapping=trace_id_mapping,
            )
            # Note: Status remains "processing" until embeddings complete

        except Exception as e:
            logger.error(
                f"[process_completed_evaluation] {log_prefix} Failed to start embedding batch | {e}",
                exc_info=True,
            )
            # Don't fail the entire evaluation, just mark as completed without embeddings
            eval_run = update_evaluation_run(
                session=session,
                eval_run=eval_run,
                status="completed",
                error_message=f"Embeddings failed: {str(e)}",
            )

        logger.info(
            f"[process_completed_evaluation] {log_prefix} Processed evaluation | items={len(results)}"
        )

        return eval_run

    except Exception as e:
        logger.error(
            f"[process_completed_evaluation] {log_prefix} Failed to process completed evaluation | {e}",
            exc_info=True,
        )
        # Mark as failed
        return update_evaluation_run(
            session=session,
            eval_run=eval_run,
            status="failed",
            error_message=f"Processing failed: {str(e)}",
        )


async def process_completed_embedding_batch(
    eval_run: EvaluationRun,
    session: Session,
    openai_client: OpenAI,
    langfuse: Langfuse,
) -> EvaluationRun:
    """
    Process a completed embedding batch and calculate similarity scores.

    This function:
    1. Downloads embedding batch results
    2. Parses embeddings (output + ground_truth pairs)
    3. Calculates cosine similarity for each pair
    4. Calculates average and statistics
    5. Updates eval_run.score with results
    6. Updates Langfuse traces with per-item cosine similarity scores
    7. Marks evaluation as completed

    Args:
        eval_run: EvaluationRun database object
        session: Database session
        openai_client: Configured OpenAI client
        langfuse: Configured Langfuse client

    Returns:
        Updated EvaluationRun object with similarity scores

    Raises:
        Exception: If processing fails
    """
    log_prefix = f"[org={eval_run.organization_id}][project={eval_run.project_id}][eval={eval_run.id}]"
    logger.info(
        f"[process_completed_embedding_batch] {log_prefix} Processing completed embedding batch"
    )

    try:
        # Step 1: Get embedding_batch_job
        if not eval_run.embedding_batch_job_id:
            raise ValueError(
                f"EvaluationRun {eval_run.id} has no embedding_batch_job_id"
            )

        embedding_batch_job = get_batch_job(
            session=session, batch_job_id=eval_run.embedding_batch_job_id
        )
        if not embedding_batch_job:
            raise ValueError(
                f"Embedding BatchJob {eval_run.embedding_batch_job_id} not found for evaluation {eval_run.id}"
            )

        # Step 2: Create provider and download results
        provider = OpenAIBatchProvider(client=openai_client)
        raw_results = download_batch_results(
            provider=provider, batch_job=embedding_batch_job
        )

        # Step 3: Parse embedding results
        embedding_pairs = parse_embedding_results(raw_results=raw_results)

        if not embedding_pairs:
            raise ValueError("No valid embedding pairs found in batch output")

        # Step 4: Calculate similarity scores
        similarity_stats = calculate_average_similarity(embedding_pairs=embedding_pairs)

        # Step 5: Update evaluation_run with scores in summary_scores format
        # This format is consistent with what Langfuse returns when fetching traces
        eval_run.score = {
            "summary_scores": [
                {
                    "name": "cosine_similarity",
                    "avg": round(float(similarity_stats["cosine_similarity_avg"]), 2),
                    "std": round(float(similarity_stats["cosine_similarity_std"]), 2),
                    "total_pairs": similarity_stats["total_pairs"],
                    "data_type": "NUMERIC",
                }
            ]
        }

        # Step 6: Update Langfuse traces with cosine similarity scores
        logger.info(
            f"[process_completed_embedding_batch] {log_prefix} Updating Langfuse traces with cosine similarity scores"
        )
        per_item_scores = similarity_stats.get("per_item_scores", [])
        if per_item_scores:
            try:
                update_traces_with_cosine_scores(
                    langfuse=langfuse,
                    per_item_scores=per_item_scores,
                )
            except Exception as e:
                # Log error but don't fail the evaluation
                logger.error(
                    f"[process_completed_embedding_batch] {log_prefix} Failed to update Langfuse traces with scores | {e}",
                    exc_info=True,
                )

        # Step 7: Mark evaluation as completed
        eval_run = update_evaluation_run(
            session=session, eval_run=eval_run, status="completed", score=eval_run.score
        )

        logger.info(
            f"[process_completed_embedding_batch] {log_prefix} Completed evaluation | avg_similarity={similarity_stats['cosine_similarity_avg']:.3f}"
        )

        return eval_run

    except Exception as e:
        logger.error(
            f"[process_completed_embedding_batch] {log_prefix} Failed to process completed embedding batch | {e}",
            exc_info=True,
        )
        # Mark as completed anyway, but with error message
        return update_evaluation_run(
            session=session,
            eval_run=eval_run,
            status="completed",
            error_message=f"Embedding processing failed: {str(e)}",
        )


async def check_and_process_evaluation(
    eval_run: EvaluationRun,
    session: Session,
    openai_client: OpenAI,
    langfuse: Langfuse,
) -> dict[str, Any]:
    """
    Check evaluation batch status and process if completed.

    This function handles both the response batch and embedding batch:
    1. If embedding_batch_job_id exists, checks and processes embedding batch first
    2. Otherwise, checks and processes the main response batch
    3. Triggers appropriate processing based on batch completion status

    Args:
        eval_run: EvaluationRun database object
        session: Database session
        openai_client: Configured OpenAI client
        langfuse: Configured Langfuse client

    Returns:
        Dict with status information:
        {
            "run_id": 123,
            "run_name": "test_run",
            "previous_status": "processing",
            "current_status": "completed",
            "batch_status": "completed",
            "action": "processed" | "embeddings_completed" | "embeddings_failed" | "failed" | "no_change"
        }
    """
    log_prefix = f"[org={eval_run.organization_id}][project={eval_run.project_id}][eval={eval_run.id}]"
    previous_status = eval_run.status

    try:
        # Check if we need to process embedding batch first
        if eval_run.embedding_batch_job_id and eval_run.status == "processing":
            embedding_batch_job = get_batch_job(
                session=session, batch_job_id=eval_run.embedding_batch_job_id
            )

            if embedding_batch_job:
                # Poll embedding batch status
                provider = OpenAIBatchProvider(client=openai_client)
                poll_batch_status(
                    session=session, provider=provider, batch_job=embedding_batch_job
                )
                session.refresh(embedding_batch_job)

                embedding_status = embedding_batch_job.provider_status

                if embedding_status == "completed":
                    logger.info(
                        f"[check_and_process_evaluation] {log_prefix} Processing embedding batch | provider_batch_id={embedding_batch_job.provider_batch_id}"
                    )

                    await process_completed_embedding_batch(
                        eval_run=eval_run,
                        session=session,
                        openai_client=openai_client,
                        langfuse=langfuse,
                    )

                    return {
                        "run_id": eval_run.id,
                        "run_name": eval_run.run_name,
                        "previous_status": previous_status,
                        "current_status": eval_run.status,
                        "provider_status": embedding_status,
                        "action": "embeddings_completed",
                    }

                elif embedding_status in ["failed", "expired", "cancelled"]:
                    logger.error(
                        f"[check_and_process_evaluation] {log_prefix} Embedding batch failed | provider_batch_id={embedding_batch_job.provider_batch_id} | {embedding_batch_job.error_message}"
                    )
                    # Mark as completed without embeddings
                    eval_run = update_evaluation_run(
                        session=session,
                        eval_run=eval_run,
                        status="completed",
                        error_message=f"Embedding batch failed: {embedding_batch_job.error_message}",
                    )

                    return {
                        "run_id": eval_run.id,
                        "run_name": eval_run.run_name,
                        "previous_status": previous_status,
                        "current_status": "completed",
                        "provider_status": embedding_status,
                        "action": "embeddings_failed",
                    }

                else:
                    # Embedding batch still processing
                    return {
                        "run_id": eval_run.id,
                        "run_name": eval_run.run_name,
                        "previous_status": previous_status,
                        "current_status": eval_run.status,
                        "provider_status": embedding_status,
                        "action": "no_change",
                    }

        # Get batch_job (main response batch)
        if not eval_run.batch_job_id:
            raise ValueError(f"EvaluationRun {eval_run.id} has no batch_job_id")

        batch_job = get_batch_job(session=session, batch_job_id=eval_run.batch_job_id)
        if not batch_job:
            raise ValueError(
                f"BatchJob {eval_run.batch_job_id} not found for evaluation {eval_run.id}"
            )

        # IMPORTANT: Poll OpenAI to get the latest status before checking
        provider = OpenAIBatchProvider(client=openai_client)
        poll_batch_status(session=session, provider=provider, batch_job=batch_job)

        # Refresh batch_job to get the updated provider_status
        session.refresh(batch_job)
        provider_status = batch_job.provider_status

        # Handle different provider statuses
        if provider_status == "completed":
            # Process the completed evaluation
            await process_completed_evaluation(
                eval_run=eval_run,
                session=session,
                openai_client=openai_client,
                langfuse=langfuse,
            )

            return {
                "run_id": eval_run.id,
                "run_name": eval_run.run_name,
                "previous_status": previous_status,
                "current_status": eval_run.status,
                "provider_status": provider_status,
                "action": "processed",
            }

        elif provider_status in ["failed", "expired", "cancelled"]:
            # Mark evaluation as failed based on provider status
            error_msg = batch_job.error_message or f"Provider batch {provider_status}"

            eval_run = update_evaluation_run(
                session=session,
                eval_run=eval_run,
                status="failed",
                error_message=error_msg,
            )

            logger.error(
                f"[check_and_process_evaluation] {log_prefix} Batch failed | provider_batch_id={batch_job.provider_batch_id} | {error_msg}"
            )

            return {
                "run_id": eval_run.id,
                "run_name": eval_run.run_name,
                "previous_status": previous_status,
                "current_status": "failed",
                "provider_status": provider_status,
                "action": "failed",
                "error": error_msg,
            }

        else:
            # Still in progress (validating, in_progress, finalizing)
            return {
                "run_id": eval_run.id,
                "run_name": eval_run.run_name,
                "previous_status": previous_status,
                "current_status": eval_run.status,
                "provider_status": provider_status,
                "action": "no_change",
            }

    except Exception as e:
        logger.error(
            f"[check_and_process_evaluation] {log_prefix} Error checking evaluation | {e}",
            exc_info=True,
        )

        # Mark as failed
        update_evaluation_run(
            session=session,
            eval_run=eval_run,
            status="failed",
            error_message=f"Checking failed: {str(e)}",
        )

        return {
            "run_id": eval_run.id,
            "run_name": eval_run.run_name,
            "previous_status": previous_status,
            "current_status": "failed",
            "provider_status": "unknown",
            "action": "failed",
            "error": str(e),
        }


async def poll_all_pending_evaluations(session: Session, org_id: int) -> dict[str, Any]:
    """
    Poll all pending evaluations for an organization.

    Args:
        session: Database session
        org_id: Organization ID

    Returns:
        Summary dict:
        {
            "total": 5,
            "processed": 2,
            "failed": 1,
            "still_processing": 2,
            "details": [...]
        }
    """
    # Get pending evaluations (status = "processing")
    statement = select(EvaluationRun).where(
        EvaluationRun.status == "processing",
        EvaluationRun.organization_id == org_id,
    )
    pending_runs = session.exec(statement).all()

    if not pending_runs:
        return {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "still_processing": 0,
            "details": [],
        }
    # Group evaluations by project_id since credentials are per project
    evaluations_by_project = defaultdict(list)
    for run in pending_runs:
        evaluations_by_project[run.project_id].append(run)

    # Process each project separately
    all_results = []
    total_processed_count = 0
    total_failed_count = 0
    total_still_processing_count = 0

    for project_id, project_runs in evaluations_by_project.items():
        try:
            # Get API clients for this project
            try:
                openai_client = get_openai_client(
                    session=session,
                    org_id=org_id,
                    project_id=project_id,
                )
                langfuse = get_langfuse_client(
                    session=session,
                    org_id=org_id,
                    project_id=project_id,
                )
            except HTTPException as http_exc:
                logger.error(
                    f"[poll_all_pending_evaluations] Failed to get API clients | org_id={org_id} | project_id={project_id} | error={http_exc.detail}"
                )
                # Mark all runs in this project as failed due to client configuration error
                for eval_run in project_runs:
                    # Persist failure status to database
                    update_evaluation_run(
                        session=session,
                        eval_run=eval_run,
                        status="failed",
                        error_message=http_exc.detail,
                    )

                    all_results.append(
                        {
                            "run_id": eval_run.id,
                            "run_name": eval_run.run_name,
                            "action": "failed",
                            "error": http_exc.detail,
                        }
                    )
                    total_failed_count += 1
                continue

            # Process each evaluation in this project
            for eval_run in project_runs:
                try:
                    result = await check_and_process_evaluation(
                        eval_run=eval_run,
                        session=session,
                        openai_client=openai_client,
                        langfuse=langfuse,
                    )
                    all_results.append(result)

                    if result["action"] == "processed":
                        total_processed_count += 1
                    elif result["action"] == "failed":
                        total_failed_count += 1
                    else:
                        total_still_processing_count += 1

                except Exception as e:
                    logger.error(
                        f"[poll_all_pending_evaluations] Failed to check evaluation run | run_id={eval_run.id} | {e}",
                        exc_info=True,
                    )
                    # Persist failure status to database
                    update_evaluation_run(
                        session=session,
                        eval_run=eval_run,
                        status="failed",
                        error_message=f"Check failed: {str(e)}",
                    )

                    all_results.append(
                        {
                            "run_id": eval_run.id,
                            "run_name": eval_run.run_name,
                            "action": "failed",
                            "error": str(e),
                        }
                    )
                    total_failed_count += 1

        except Exception as e:
            logger.error(
                f"[poll_all_pending_evaluations] Failed to process project | project_id={project_id} | {e}",
                exc_info=True,
            )
            # Mark all runs in this project as failed
            for eval_run in project_runs:
                # Persist failure status to database
                update_evaluation_run(
                    session=session,
                    eval_run=eval_run,
                    status="failed",
                    error_message=f"Project processing failed: {str(e)}",
                )

                all_results.append(
                    {
                        "run_id": eval_run.id,
                        "run_name": eval_run.run_name,
                        "action": "failed",
                        "error": f"Project processing failed: {str(e)}",
                    }
                )
                total_failed_count += 1

    summary = {
        "total": len(pending_runs),
        "processed": total_processed_count,
        "failed": total_failed_count,
        "still_processing": total_still_processing_count,
        "details": all_results,
    }

    logger.info(
        f"[poll_all_pending_evaluations] Polling summary | org_id={org_id} | processed={total_processed_count} | failed={total_failed_count} | still_processing={total_still_processing_count}"
    )

    return summary
