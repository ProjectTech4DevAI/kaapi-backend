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
from sqlmodel import Session

from app.core.batch import (
    BATCH_KEY,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    start_batch_job,
)
from app.core.batch.client import GeminiClient
from app.models import EvaluationRun
from app.models.batch_job import BatchJobType
from app.services.llm.mappers import (
    map_kaapi_to_google_params,
    map_kaapi_to_openai_params,
)
from app.services.llm.providers.registry import LLMProvider
from app.utils import get_openai_client

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
        logger.warning(
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


def reconcile_tracing_client(
    session: Session,
    eval_run: EvaluationRun,
    langfuse: Langfuse | None,
) -> Langfuse | None:
    """Return the live client only if the run's dataset is Langfuse-backed, else
    None, so an opt-out dataset is never traced even if the flag is later on."""
    if langfuse is None:
        return None

    from app.crud.evaluations.dataset import get_dataset_by_id

    dataset = get_dataset_by_id(
        session=session,
        dataset_id=eval_run.dataset_id,
        organization_id=eval_run.organization_id,
        project_id=eval_run.project_id,
    )
    return langfuse if (dataset and dataset.langfuse_dataset_id) else None


def load_evaluation_dataset_items(
    session: Session,
    eval_run: EvaluationRun,
    langfuse: Langfuse | None,
) -> list[dict[str, Any]]:
    """Load dataset items, sourced by the dataset's backing (Langfuse if
    langfuse_dataset_id is set, else object store) so ids stay stable across an
    eval's build and process even if the tracing flag is toggled in between."""
    # Lazy imports avoid a crud -> services -> crud import cycle.
    from app.core.cloud import get_cloud_storage
    from app.crud.evaluations.dataset import (
        download_csv_from_object_store,
        get_dataset_by_id,
    )
    from app.crud.evaluations.score import DEFAULT_CATEGORY
    from app.services.evaluations.validators import parse_csv_items

    dataset = get_dataset_by_id(
        session=session,
        dataset_id=eval_run.dataset_id,
        organization_id=eval_run.organization_id,
        project_id=eval_run.project_id,
    )
    if not dataset:
        raise ValueError(
            f"Cannot source dataset items for eval_run {eval_run.id}: dataset "
            f"{eval_run.dataset_id} not found"
        )

    # Langfuse only when tracing is on AND the dataset is Langfuse-backed;
    # otherwise fall back to object store (opt-out, or a Langfuse-backed dataset
    # run with tracing off).
    if langfuse is not None and dataset.langfuse_dataset_id:
        return fetch_dataset_items(
            langfuse=langfuse, dataset_name=eval_run.dataset_name
        )

    if not dataset.object_store_url:
        raise ValueError(
            f"Cannot source dataset items for eval_run {eval_run.id}: dataset "
            f"{eval_run.dataset_id} has no object-store backing"
        )

    storage = get_cloud_storage(session=session, project_id=eval_run.project_id)
    csv_content = download_csv_from_object_store(
        storage=storage, object_store_url=dataset.object_store_url
    )
    original_items = parse_csv_items(csv_content)
    # JSONB may hand back a str/float; coerce and floor at 1 to keep range() safe.
    duplication_factor = max(
        1, int((dataset.dataset_metadata or {}).get("duplication_factor", 1))
    )

    items: list[dict[str, Any]] = []
    for row_idx, item in enumerate(original_items):
        for dup_idx in range(duplication_factor):
            items.append(
                {
                    "id": f"item_{row_idx}_{dup_idx}",
                    "input": {"question": item["question"]},
                    "expected_output": {"answer": item["answer"]},
                    "metadata": {
                        "category": item.get("category") or DEFAULT_CATEGORY,
                        "question_id": f"item_{row_idx}",
                    },
                }
            )
    return items


def build_openai_evaluation_jsonl(
    dataset_items: list[dict[str, Any]], openai_params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build OpenAI Responses API batch JSONL from Langfuse dataset items.

    Each line:
    {
        BATCH_KEY: <dataset_item_id>,
        "method": "POST",
        "url": "/v1/responses",
        "body": { ...openai_params, "input": <question> }
    }
    """
    jsonl_data: list[dict[str, Any]] = []
    for item in dataset_items:
        question = item["input"].get("question", "")
        if not question:
            logger.warning(
                f"[build_openai_evaluation_jsonl] Skipping item - no question found | item_id={item['id']}"
            )
            continue

        body = dict(openai_params)
        body["input"] = question

        jsonl_data.append(
            {
                BATCH_KEY: item["id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
    return jsonl_data


def build_google_evaluation_jsonl(
    dataset_items: list[dict[str, Any]], google_params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build Gemini batch JSONL from Langfuse dataset items.

    Each line:
    {
        "key": <dataset_item_id>,
        "request": { contents, systemInstruction?, generationConfig? }
    }
    """
    jsonl_data: list[dict[str, Any]] = []
    system_instruction = google_params.get("instructions")

    generation_config: dict[str, Any] = {}
    temperature = google_params.get("temperature")
    if temperature is not None:
        generation_config["temperature"] = temperature
    reasoning = google_params.get("reasoning")
    if reasoning:
        generation_config["thinkingConfig"] = {
            "includeThoughts": False,
            "thinkingLevel": reasoning,
        }

    for item in dataset_items:
        question = item["input"].get("question", "")
        if not question:
            logger.warning(
                f"[build_google_evaluation_jsonl] Skipping item - no question found | item_id={item['id']}"
            )
            continue

        request: dict[str, Any] = {
            "contents": [{"parts": [{"text": question}], "role": "user"}],
        }
        if system_instruction:
            request["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if generation_config:
            request["generationConfig"] = generation_config

        jsonl_data.append({"key": item["id"], "request": request})

    return jsonl_data


def start_evaluation_batch(
    langfuse: Langfuse | None,
    session: Session,
    eval_run: EvaluationRun,
    params: dict[str, Any],
    provider: str,
) -> EvaluationRun:
    """
    Fetch dataset, build JSONL, submit batch via the appropriate provider.

    Args:
        langfuse: Configured Langfuse client
        session: Database session
        eval_run: EvaluationRun database object
        params: Kaapi-standardized completion params (dict)
        provider: Completion provider ("openai" or "google-aistudio", with optional "-native" suffix)

    Returns:
        Updated EvaluationRun with batch_job_id populated
    """
    try:
        logger.info(
            f"[start_evaluation_batch] Starting evaluation batch | run={eval_run.run_name} | provider={provider}"
        )
        dataset_items = load_evaluation_dataset_items(
            session=session, eval_run=eval_run, langfuse=langfuse
        )

        base_provider = provider.replace("-native", "")

        if base_provider == LLMProvider.OPENAI:
            mapped_params, warnings = map_kaapi_to_openai_params(
                session=session, kaapi_params=params
            )
            if warnings:
                logger.info("[start_evaluation_batch] Mapper warnings: %s", warnings)

            jsonl_data = build_openai_evaluation_jsonl(
                dataset_items=dataset_items, openai_params=mapped_params
            )
            if not jsonl_data:
                raise ValueError(
                    "Evaluation dataset did not produce any JSONL entries (missing questions?)."
                )

            openai_client = get_openai_client(
                session=session,
                org_id=eval_run.organization_id,
                project_id=eval_run.project_id,
            )
            batch_provider = OpenAIBatchProvider(client=openai_client)

            batch_config = {
                "endpoint": "/v1/responses",
                "description": f"Evaluation: {eval_run.run_name}",
                "completion_window": "24h",
                "evaluation_config": params,
            }

            batch_job = start_batch_job(
                session=session,
                provider=batch_provider,
                provider_name="openai",
                job_type=BatchJobType.EVALUATION,
                organization_id=eval_run.organization_id,
                project_id=eval_run.project_id,
                jsonl_data=jsonl_data,
                config=batch_config,
            )

        elif base_provider == LLMProvider.GOOGLE_AISTUDIO:
            mapped_params, warnings = map_kaapi_to_google_params(
                kaapi_params=params, completion_type="text"
            )
            if warnings:
                logger.info("[start_evaluation_batch] Mapper warnings: %s", warnings)

            jsonl_data = build_google_evaluation_jsonl(
                dataset_items=dataset_items, google_params=mapped_params
            )
            if not jsonl_data:
                raise ValueError(
                    "Evaluation dataset did not produce any JSONL entries (missing questions?)."
                )

            gemini_client = GeminiClient.from_credentials(
                session=session,
                org_id=eval_run.organization_id,
                project_id=eval_run.project_id,
            )
            model_name = mapped_params.get("model", "gemini-2.5-pro")
            batch_provider = GeminiBatchProvider(
                client=gemini_client.client, model=model_name
            )

            batch_config = {
                "display_name": f"evaluation-{eval_run.run_name}",
                "model": f"models/{model_name}",
            }

            batch_job = start_batch_job(
                session=session,
                provider=batch_provider,
                provider_name="google-aistudio",
                job_type=BatchJobType.EVALUATION,
                organization_id=eval_run.organization_id,
                project_id=eval_run.project_id,
                jsonl_data=jsonl_data,
                config=batch_config,
            )

        else:
            raise ValueError(f"Unsupported provider for evaluation batches: {provider}")

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
            f"run={eval_run.run_name} | items={batch_job.total_items} | "
            f"provider={base_provider}"
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
