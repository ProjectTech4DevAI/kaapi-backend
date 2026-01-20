"""
Evaluation-specific batch preparation and orchestration.

This module handles:
1. Fetching dataset items from Langfuse
2. Building evaluation-specific JSONL for batch processing
3. Starting evaluation batches using generic batch infrastructure
"""

import logging
from typing import Any

from langfuse import Langfuse
from openai import OpenAI
from sqlmodel import Session

from app.core.batch import OpenAIBatchProvider, start_batch_job
from app.models import EvaluationRun

logger = logging.getLogger(__name__)


def fetch_dataset_items(langfuse: Langfuse, dataset_name: str) -> list[dict[str, Any]]:
    """
    Fetch all items from a Langfuse dataset.

    Args:
        langfuse: Configured Langfuse client
        dataset_name: Name of the dataset to fetch

    Returns:
        List of dataset items with input and expected_output

    Raises:
        ValueError: If dataset not found or empty
    """
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        logger.error(
            f"[fetch_dataset_items] Failed to fetch dataset | dataset={dataset_name} | {e}"
        )
        raise ValueError(f"Dataset '{dataset_name}' not found: {e}")

    if not dataset.items:
        raise ValueError(f"Dataset '{dataset_name}' is empty")

    items = []
    for item in dataset.items:
        items.append(
            {
                "id": item.id,
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata if hasattr(item, "metadata") else {},
            }
        )
    return items


def build_evaluation_jsonl(
    dataset_items: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Build JSONL data for evaluation batch using OpenAI Responses API.

    Each line is a dict with:
    - custom_id: Unique identifier for the request (dataset item ID)
    - method: POST
    - url: /v1/responses
    - body: Response request using config as-is with input from dataset

    Args:
        dataset_items: List of dataset items from Langfuse
        config: Evaluation configuration dict with OpenAI Responses API parameters.
            This config is used as-is in the body, with only "input" being added
            from the dataset. Config can include any fields like:
            - model (required)
            - instructions
            - tools
            - reasoning
            - text
            - temperature
            - include
            etc.

    Returns:
        List of dictionaries (JSONL data)
    """
    jsonl_data = []

    for item in dataset_items:
        # Extract question from input
        question = item["input"].get("question", "")
        if not question:
            logger.warning(
                f"[build_evaluation_jsonl] Skipping item - no question found | item_id={item['id']}"
            )
            continue

        # Build the batch request object for Responses API
        # Use config as-is and only add the input field
        batch_request = {
            "custom_id": item["id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                **config,  # Use config as-is
                "input": question,  # Add input from dataset
            },
        }

        jsonl_data.append(batch_request)
    return jsonl_data


def start_evaluation_batch(
    langfuse: Langfuse,
    openai_client: OpenAI,
    session: Session,
    eval_run: EvaluationRun,
    config: dict[str, Any],
) -> EvaluationRun:
    """
    Fetch data, build JSONL, and start evaluation batch.

    This function orchestrates the evaluation-specific logic and delegates
    to the generic batch infrastructure for actual batch creation.

    Args:
        langfuse: Configured Langfuse client
        openai_client: Configured OpenAI client
        session: Database session
        eval_run: EvaluationRun database object (with run_name, dataset_name, config)
        config: Evaluation configuration dict with llm, instructions, vector_store_ids

    Returns:
        Updated EvaluationRun with batch_job_id populated

    Raises:
        Exception: If any step fails
    """
    try:
        # Step 1: Fetch dataset items from Langfuse
        logger.info(
            f"[start_evaluation_batch] Starting evaluation batch | run={eval_run.run_name}"
        )
        dataset_items = fetch_dataset_items(
            langfuse=langfuse, dataset_name=eval_run.dataset_name
        )

        # Step 2: Build evaluation-specific JSONL
        jsonl_data = build_evaluation_jsonl(dataset_items=dataset_items, config=config)

        if not jsonl_data:
            raise ValueError(
                "Evaluation dataset did not produce any JSONL entries (missing questions?)."
            )

        # Step 3: Create batch provider
        provider = OpenAIBatchProvider(client=openai_client)

        # Step 4: Prepare batch configuration
        batch_config = {
            "endpoint": "/v1/responses",
            "description": f"Evaluation: {eval_run.run_name}",
            "completion_window": "24h",
            # Store complete config for reference
            "evaluation_config": config,
        }

        # Step 5: Start batch job using generic infrastructure
        batch_job = start_batch_job(
            session=session,
            provider=provider,
            provider_name="openai",
            job_type="evaluation",
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
            jsonl_data=jsonl_data,
            config=batch_config,
        )

        # Step 6: Link batch_job to evaluation_run
        eval_run.batch_job_id = batch_job.id
        eval_run.status = "processing"
        eval_run.total_items = batch_job.total_items

        session.add(eval_run)
        session.commit()
        session.refresh(eval_run)

        logger.info(
            f"[start_evaluation_batch] Successfully started evaluation batch | "
            f"batch_job_id={batch_job.id} | "
            f"provider_batch_id={batch_job.provider_batch_id} | "
            f"run={eval_run.run_name} | items={batch_job.total_items}"
        )

        return eval_run

    except Exception as e:
        logger.error(
            f"[start_evaluation_batch] Failed to start evaluation batch | {e}",
            exc_info=True,
        )
        eval_run.status = "failed"
        eval_run.error_message = str(e)
        session.add(eval_run)
        session.commit()
        raise
