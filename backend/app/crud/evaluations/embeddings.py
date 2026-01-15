"""
Embedding-based similarity scoring for evaluation runs.

This module handles:
1. Building JSONL for embedding batch requests
2. Parsing embedding results from batch API
3. Calculating cosine similarity between embeddings
4. Orchestrating embedding batch creation and processing
"""

import logging
from typing import Any

import numpy as np
from openai import OpenAI
from sqlmodel import Session

from app.core.batch import OpenAIBatchProvider, start_batch_job
from app.core.util import now
from app.models import EvaluationRun

logger = logging.getLogger(__name__)

# Valid embedding models with their dimensions
VALID_EMBEDDING_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def validate_embedding_model(model: str) -> None:
    """
    Validate that the embedding model is supported.

    Args:
        model: The embedding model name

    Raises:
        ValueError: If the model is not supported
    """
    if model not in VALID_EMBEDDING_MODELS:
        valid_models = ", ".join(VALID_EMBEDDING_MODELS.keys())
        raise ValueError(
            f"Invalid embedding model '{model}'. Supported models: {valid_models}"
        )


def build_embedding_jsonl(
    results: list[dict[str, Any]],
    trace_id_mapping: dict[str, str],
    embedding_model: str = "text-embedding-3-large",
) -> list[dict[str, Any]]:
    """
    Build JSONL data for embedding batch using OpenAI Embeddings API.

    Each line is a dict with:
    - custom_id: Langfuse trace_id (for direct score updates)
    - method: POST
    - url: /v1/embeddings
    - body: Embedding request with input array [output, ground_truth]

    Args:
        results: List of evaluation results from parse_evaluation_output()
                 Format: [
                     {
                         "item_id": "item_123",
                         "question": "What is 2+2?",
                         "generated_output": "The answer is 4",
                         "ground_truth": "4"
                     },
                     ...
                 ]
        trace_id_mapping: Mapping of item_id to Langfuse trace_id
        embedding_model: OpenAI embedding model to use (default: text-embedding-3-large)

    Returns:
        List of dictionaries (JSONL data)
    """
    # Validate embedding model
    validate_embedding_model(embedding_model)

    logger.info(
        f"[build_embedding_jsonl] Building JSONL | items={len(results)} | model={embedding_model}"
    )

    jsonl_data = []

    for result in results:
        item_id = result.get("item_id")
        generated_output = result.get("generated_output", "")
        ground_truth = result.get("ground_truth", "")

        if not item_id:
            logger.warning("Skipping result with no item_id")
            continue

        # Get trace_id from mapping
        trace_id = trace_id_mapping.get(item_id)
        if not trace_id:
            logger.warning(f"Skipping item {item_id} - no trace_id found")
            continue

        # Skip if either output or ground_truth is empty
        if not generated_output or not ground_truth:
            logger.warning(f"Skipping item {item_id} - empty output or ground_truth")
            continue

        # Build the batch request object for Embeddings API
        # Use trace_id as custom_id for direct score updates
        batch_request = {
            "custom_id": trace_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": embedding_model,
                "input": [
                    generated_output,  # Index 0
                    ground_truth,  # Index 1
                ],
                "encoding_format": "float",
            },
        }

        jsonl_data.append(batch_request)

    logger.info(f"Built {len(jsonl_data)} embedding JSONL lines")
    return jsonl_data


def parse_embedding_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Parse embedding batch output into structured embedding pairs.

    Args:
        raw_results: Raw results from batch provider (list of JSONL lines)

    Returns:
        List of embedding pairs in format:
        [
            {
                "trace_id": "trace-uuid-123",
                "output_embedding": [0.1, 0.2, ...],
                "ground_truth_embedding": [0.15, 0.22, ...]
            },
            ...
        ]
    """
    logger.info(f"Parsing embedding results from {len(raw_results)} lines")

    embedding_pairs = []

    for line_num, response in enumerate(raw_results, 1):
        try:
            # Extract custom_id (which is now the Langfuse trace_id)
            trace_id = response.get("custom_id")
            if not trace_id:
                logger.warning(f"Line {line_num}: No custom_id found, skipping")
                continue

            # Handle errors in batch processing
            if response.get("error"):
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"Trace {trace_id} had error: {error_msg}")
                continue

            # Extract the response body
            response_body = response.get("response", {}).get("body", {})
            embedding_data = response_body.get("data", [])

            if len(embedding_data) < 2:
                logger.warning(
                    f"Trace {trace_id}: Expected 2 embeddings, got {len(embedding_data)}"
                )
                continue

            # Extract embeddings by index
            # Index 0 = generated_output embedding
            # Index 1 = ground_truth embedding
            output_embedding = None
            ground_truth_embedding = None

            for emb_obj in embedding_data:
                index = emb_obj.get("index")
                embedding = emb_obj.get("embedding")

                if embedding is None:
                    continue

                if index == 0:
                    output_embedding = embedding
                elif index == 1:
                    ground_truth_embedding = embedding

            if output_embedding is None or ground_truth_embedding is None:
                logger.warning(
                    f"Trace {trace_id}: Missing embeddings (output={output_embedding is not None}, "
                    f"ground_truth={ground_truth_embedding is not None})"
                )
                continue

            embedding_pairs.append(
                {
                    "trace_id": trace_id,
                    "output_embedding": output_embedding,
                    "ground_truth_embedding": ground_truth_embedding,
                }
            )

        except Exception as e:
            logger.error(f"Line {line_num}: Unexpected error: {e}", exc_info=True)
            continue

    logger.info(
        f"Parsed {len(embedding_pairs)} embedding pairs from {len(raw_results)} lines"
    )
    return embedding_pairs


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors using numpy.

    Formula: similarity = dot(vec1, vec2) / (||vec1|| * ||vec2||)

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (range: -1 to 1, typically 0 to 1 for embeddings)
    """
    # Convert to numpy arrays
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate norms
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Handle edge case of zero vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (norm_v1 * norm_v2)

    return float(similarity)


def calculate_average_similarity(
    embedding_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate cosine similarity statistics for all embedding pairs.

    Args:
        embedding_pairs: List of embedding pairs from parse_embedding_results()

    Returns:
        Dictionary with similarity statistics:
        {
            "cosine_similarity_avg": 0.87,
            "cosine_similarity_std": 0.12,
            "total_pairs": 50,
            "per_item_scores": [...]  # Individual scores with trace_ids
        }
    """
    logger.info(f"Calculating similarity for {len(embedding_pairs)} pairs")

    if not embedding_pairs:
        return {
            "cosine_similarity_avg": 0.0,
            "cosine_similarity_std": 0.0,
            "total_pairs": 0,
            "per_item_scores": [],
        }

    similarities = []
    per_item_scores = []

    for pair in embedding_pairs:
        try:
            output_emb = pair["output_embedding"]
            ground_truth_emb = pair["ground_truth_embedding"]

            similarity = calculate_cosine_similarity(output_emb, ground_truth_emb)
            similarities.append(similarity)

            per_item_scores.append(
                {
                    "trace_id": pair["trace_id"],
                    "cosine_similarity": similarity,
                }
            )

        except Exception as e:
            logger.error(
                f"Error calculating similarity for trace {pair.get('trace_id')}: {e}"
            )
            continue

    if not similarities:
        logger.warning("No valid similarities calculated")
        return {
            "cosine_similarity_avg": 0.0,
            "cosine_similarity_std": 0.0,
            "total_pairs": 0,
            "per_item_scores": [],
        }

    # Calculate statistics
    similarities_array = np.array(similarities)

    stats = {
        "cosine_similarity_avg": float(np.mean(similarities_array)),
        "cosine_similarity_std": float(np.std(similarities_array)),
        "total_pairs": len(similarities),
        "per_item_scores": per_item_scores,
    }

    logger.info(
        f"Calculated similarity stats: avg={stats['cosine_similarity_avg']:.3f}, "
        f"std={stats['cosine_similarity_std']:.3f}"
    )

    return stats


def start_embedding_batch(
    session: Session,
    openai_client: OpenAI,
    eval_run: EvaluationRun,
    results: list[dict[str, Any]],
    trace_id_mapping: dict[str, str],
) -> EvaluationRun:
    """
    Start embedding batch for similarity scoring.

    This function orchestrates the embedding batch creation:
    1. Builds embedding JSONL from evaluation results with trace_ids
    2. Creates batch via generic infrastructure (job_type="embedding")
    3. Links embedding_batch_job_id to eval_run
    4. Keeps status as "processing"

    Args:
        session: Database session
        openai_client: Configured OpenAI client
        eval_run: EvaluationRun database object
        results: Parsed evaluation results (output + ground_truth pairs)
        trace_id_mapping: Mapping of item_id to Langfuse trace_id

    Returns:
        Updated EvaluationRun with embedding_batch_job_id populated

    Raises:
        Exception: If any step fails
    """
    try:
        logger.info(f"Starting embedding batch for evaluation run {eval_run.id}")

        # Get embedding model from config (default: text-embedding-3-large)
        embedding_model = eval_run.config.get(
            "embedding_model", "text-embedding-3-large"
        )

        # Validate and fallback to default if invalid
        try:
            validate_embedding_model(embedding_model)
        except ValueError as e:
            logger.warning(
                f"Invalid embedding model '{embedding_model}' in config: {e}. "
                f"Falling back to text-embedding-3-large"
            )
            embedding_model = "text-embedding-3-large"

        # Step 1: Build embedding JSONL with trace_ids
        jsonl_data = build_embedding_jsonl(
            results=results,
            trace_id_mapping=trace_id_mapping,
            embedding_model=embedding_model,
        )

        if not jsonl_data:
            raise ValueError("No valid items to create embeddings for")

        # Step 2: Create batch provider
        provider = OpenAIBatchProvider(client=openai_client)

        # Step 3: Prepare batch configuration
        batch_config = {
            "endpoint": "/v1/embeddings",
            "description": f"Embeddings for evaluation: {eval_run.run_name}",
            "completion_window": "24h",
            "embedding_model": embedding_model,
        }

        # Step 4: Start batch job using generic infrastructure
        batch_job = start_batch_job(
            session=session,
            provider=provider,
            provider_name="openai",
            job_type="embedding",
            organization_id=eval_run.organization_id,
            project_id=eval_run.project_id,
            jsonl_data=jsonl_data,
            config=batch_config,
        )

        # Step 5: Link embedding_batch_job to evaluation_run
        eval_run.embedding_batch_job_id = batch_job.id
        # Keep status as "processing" - will change to "completed" after embeddings
        eval_run.updated_at = now()

        session.add(eval_run)
        session.commit()
        session.refresh(eval_run)

        logger.info(
            f"Successfully started embedding batch: batch_job_id={batch_job.id}, "
            f"provider_batch_id={batch_job.provider_batch_id} "
            f"for evaluation run {eval_run.id} with {batch_job.total_items} items"
        )

        return eval_run

    except Exception as e:
        logger.error(f"Failed to start embedding batch: {e}", exc_info=True)
        # Don't update eval_run status here - let caller decide
        raise
