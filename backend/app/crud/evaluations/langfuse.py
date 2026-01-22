"""
Langfuse integration for evaluation runs.

This module handles:
1. Creating dataset runs in Langfuse
2. Creating traces for each evaluation item
3. Uploading results to Langfuse for visualization
4. Fetching trace scores from Langfuse for results
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from langfuse import Langfuse

from app.crud.evaluations.score import EvaluationScore, TraceData, TraceScore

logger = logging.getLogger(__name__)


def create_langfuse_dataset_run(
    langfuse: Langfuse,
    dataset_name: str,
    run_name: str,
    results: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, str]:
    """
    Create a dataset run in Langfuse with traces for each evaluation item.

    This function:
    1. Gets the dataset from Langfuse (which already exists)
    2. For each result, creates a trace linked to the dataset item
    3. Creates a generation within the trace with usage/model for cost tracking
    4. Logs input (question), output (generated_output), and expected (ground_truth)
    5. Returns a mapping of item_id -> trace_id for later score updates

    Note: Cost tracking in Langfuse happens at the generation level, not trace level.
    We create a generation within each trace to enable automatic cost calculation.

    Args:
        langfuse: Configured Langfuse client
        dataset_name: Name of the dataset in Langfuse
        run_name: Name for this evaluation run
        results: List of evaluation results from parse_batch_output()
                 Format: [
                     {
                         "item_id": "item_123",
                         "question": "What is 2+2?",
                         "generated_output": "4",
                         "ground_truth": "4",
                         "response_id": "resp_0b99aadf...",
                         "usage": {
                             "input_tokens": 69,
                             "output_tokens": 258,
                             "total_tokens": 327
                         }
                     },
                     ...
                 ]
        model: Model name used for evaluation (for cost calculation by Langfuse)

    Returns:
        dict[str, str]: Mapping of item_id to Langfuse trace_id

    Raises:
        Exception: If Langfuse operations fail
    """
    logger.info(
        f"[create_langfuse_dataset_run] Creating Langfuse dataset run | "
        f"run_name={run_name} | dataset={dataset_name} | items={len(results)}"
    )

    try:
        # Get the dataset
        dataset = langfuse.get_dataset(dataset_name)
        dataset_items_map = {item.id: item for item in dataset.items}

        trace_id_mapping = {}

        # Create a trace for each result
        for result in results:
            item_id = result["item_id"]
            question = result["question"]
            generated_output = result["generated_output"]
            ground_truth = result["ground_truth"]
            response_id = result.get("response_id")
            usage_raw = result.get("usage")

            dataset_item = dataset_items_map.get(item_id)
            if not dataset_item:
                logger.warning(
                    f"[create_langfuse_dataset_run] Dataset item not found, skipping | "
                    f"item_id={item_id}"
                )
                continue

            try:
                with dataset_item.observe(run_name=run_name) as trace_id:
                    metadata = {
                        "ground_truth": ground_truth,
                        "item_id": item_id,
                    }
                    if response_id:
                        metadata["response_id"] = response_id

                    # Create trace with basic info
                    langfuse.trace(
                        id=trace_id,
                        input={"question": question},
                        output={"answer": generated_output},
                        metadata=metadata,
                    )

                    # Convert usage to Langfuse format
                    usage = None
                    if usage_raw:
                        usage = {
                            "input": usage_raw.get("input_tokens", 0),
                            "output": usage_raw.get("output_tokens", 0),
                            "total": usage_raw.get("total_tokens", 0),
                            "unit": "TOKENS",
                        }

                    # Create a generation within the trace for cost tracking
                    # Cost tracking happens at generation level, not trace level
                    if usage and model:
                        generation = langfuse.generation(
                            name="evaluation-response",
                            trace_id=trace_id,
                            input={"question": question},
                            metadata=metadata,
                        )
                        generation.end(
                            output={"answer": generated_output},
                            model=model,
                            usage=usage,
                        )

                    trace_id_mapping[item_id] = trace_id

            except Exception as e:
                logger.error(
                    f"[create_langfuse_dataset_run] Failed to create trace | "
                    f"item_id={item_id} | {e}",
                    exc_info=True,
                )
                continue

        langfuse.flush()
        logger.info(
            f"[create_langfuse_dataset_run] Created Langfuse dataset run | "
            f"run_name={run_name} | traces={len(trace_id_mapping)}"
        )

        return trace_id_mapping

    except Exception as e:
        logger.error(
            f"[create_langfuse_dataset_run] Failed to create Langfuse dataset run | "
            f"run_name={run_name} | {e}",
            exc_info=True,
        )
        raise


def update_traces_with_cosine_scores(
    langfuse: Langfuse,
    per_item_scores: list[dict[str, Any]],
) -> None:
    """
    Update Langfuse traces with cosine similarity scores.

    This function adds custom "cosine_similarity" scores to traces at the trace level,
    allowing them to be visualized in the Langfuse UI.

    Args:
        langfuse: Configured Langfuse client
        per_item_scores: List of per-item score dictionaries from
            calculate_average_similarity()
                        Format: [
                            {
                                "trace_id": "trace-uuid-123",
                                "cosine_similarity": 0.95
                            },
                            ...
                        ]

    Note:
        This function logs errors but does not raise exceptions to avoid blocking
        evaluation completion if Langfuse updates fail.
    """
    for score_item in per_item_scores:
        trace_id = score_item.get("trace_id")
        cosine_score = score_item.get("cosine_similarity")

        if not trace_id:
            logger.warning(
                "[update_traces_with_cosine_scores] "
                "Score item missing trace_id, skipping"
            )
            continue

        try:
            langfuse.score(
                trace_id=trace_id,
                name="cosine_similarity",
                value=cosine_score,
                comment=(
                    "Cosine similarity between generated output and "
                    "ground truth embeddings"
                ),
            )
        except Exception as e:
            logger.error(
                f"[update_traces_with_cosine_scores] Failed to add score | "
                f"trace_id={trace_id} | {e}",
                exc_info=True,
            )

    langfuse.flush()


def upload_dataset_to_langfuse(
    langfuse: Langfuse,
    items: list[dict[str, str]],
    dataset_name: str,
    duplication_factor: int,
) -> tuple[str, int]:
    """
    Upload a dataset to Langfuse from pre-parsed items.

    Args:
        langfuse: Configured Langfuse client
        items: List of dicts with 'question' and 'answer' keys (already validated)
        dataset_name: Name for the dataset in Langfuse
        duplication_factor: Number of times to duplicate each item

    Returns:
        Tuple of (langfuse_dataset_id, total_items_uploaded)

    Raises:
        Exception: If Langfuse operations fail
    """
    logger.info(
        f"[upload_dataset_to_langfuse] Uploading dataset to Langfuse | "
        f"dataset={dataset_name} | items={len(items)} | "
        f"duplication_factor={duplication_factor}"
    )

    def upload_item(item: dict[str, str], duplicate_num: int) -> bool:
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input={"question": item["question"]},
                expected_output={"answer": item["answer"]},
                metadata={
                    "original_question": item["question"],
                    "duplicate_number": duplicate_num + 1,
                    "duplication_factor": duplication_factor,
                },
            )
            return True
        except Exception as e:
            logger.error(
                f"[upload_dataset_to_langfuse] Failed to upload item | "
                f"duplicate={duplicate_num + 1} | "
                f"question={item['question'][:50]}... | {e}"
            )
            return False

    try:
        # Create or get dataset in Langfuse
        dataset = langfuse.create_dataset(name=dataset_name)

        upload_tasks = [
            (item, duplicate_num)
            for item in items
            for duplicate_num in range(duplication_factor)
        ]

        # Upload items concurrently using ThreadPoolExecutor
        total_uploaded = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all upload tasks and collect the futures
            futures = []
            for item, dup_num in upload_tasks:
                future = executor.submit(upload_item, item, dup_num)
                futures.append(future)

            for future in as_completed(futures):
                upload_successful = future.result()
                if upload_successful:
                    total_uploaded += 1

        # Final flush to ensure all items are uploaded
        langfuse.flush()

        langfuse_dataset_id = dataset.id

        logger.info(
            f"[upload_dataset_to_langfuse] Successfully uploaded to Langfuse | "
            f"items={total_uploaded} | dataset={dataset_name} | "
            f"id={langfuse_dataset_id}"
        )

        return langfuse_dataset_id, total_uploaded

    except Exception as e:
        logger.error(
            f"[upload_dataset_to_langfuse] Failed to upload dataset to Langfuse | "
            f"dataset={dataset_name} | {e}",
            exc_info=True,
        )
        raise


def fetch_trace_scores_from_langfuse(
    langfuse: Langfuse,
    dataset_name: str,
    run_name: str,
) -> EvaluationScore:
    """
    Fetch trace scores from Langfuse for an evaluation run.

    This function retrieves all traces and their scores from a Langfuse dataset run,
    including the original Q&A context for each trace.

    Args:
        langfuse: Configured Langfuse client
        dataset_name: Name of the dataset in Langfuse
        run_name: Name of the evaluation run

    Returns:
        Score data with per-trace scores and summary statistics:
        {
            "summary_scores": [
                {
                    "name": "cosine_similarity",
                    "avg": 0.87,
                    "std": 0.12,
                    "total_pairs": 50,
                    "data_type": "NUMERIC"
                },
                {
                    "name": "response_category",
                    "distribution": {"CORRECT": 10, "PARTIAL": 5},
                    "total_pairs": 15,
                    "data_type": "CATEGORICAL"
                }
            ],
            "traces": [
                {
                    "trace_id": "trace-uuid-123",
                    "question": "What is 2+2?",
                    "llm_answer": "4",
                    "ground_truth_answer": "4",
                    "scores": [
                        {
                            "name": "cosine_similarity",
                            "value": 0.95,
                            "data_type": "NUMERIC"
                        }
                    ]
                }
            ]
        }

    Raises:
        ValueError: If the run is not found in Langfuse
        Exception: If Langfuse API calls fail
    """
    logger.info(
        f"[fetch_trace_scores_from_langfuse] Fetching trace scores | "
        f"dataset={dataset_name} | run={run_name}"
    )

    try:
        # 1. Get dataset run with its items directly using get_run
        # This returns DatasetRunWithItems which includes dataset_run_items
        try:
            dataset_run = langfuse.api.datasets.get_run(dataset_name, run_name)
        except Exception as e:
            logger.error(
                f"[fetch_trace_scores_from_langfuse] Failed to get run | "
                f"dataset={dataset_name} | run={run_name} | error={e}"
            )
            raise ValueError(
                f"Run '{run_name}' not found in Langfuse dataset '{dataset_name}'"
            )

        logger.info(
            f"[fetch_trace_scores_from_langfuse] Found run | "
            f"run_name={run_name} | items_count={len(dataset_run.dataset_run_items)}"
        )

        # 2. Extract trace IDs from dataset run items
        trace_ids = [item.trace_id for item in dataset_run.dataset_run_items]

        logger.info(
            f"[fetch_trace_scores_from_langfuse] Found traces | count={len(trace_ids)}"
        )

        # 3. Fetch trace details with scores for each trace
        traces: list[TraceData] = []
        # Track score aggregations by name: {name: {"data_type": str, "values": list}}
        score_aggregations: dict[str, dict[str, Any]] = {}

        for trace_id in trace_ids:
            try:
                trace = langfuse.api.trace.get(trace_id)
                trace_data: TraceData = {
                    "trace_id": trace_id,
                    "question": "",
                    "llm_answer": "",
                    "ground_truth_answer": "",
                    "scores": [],
                }

                # Get question from input
                if trace.input:
                    if isinstance(trace.input, dict):
                        trace_data["question"] = trace.input.get("question", "")
                    elif isinstance(trace.input, str):
                        trace_data["question"] = trace.input

                # Get answer from output
                if trace.output:
                    if isinstance(trace.output, dict):
                        trace_data["llm_answer"] = trace.output.get("answer", "")
                    elif isinstance(trace.output, str):
                        trace_data["llm_answer"] = trace.output

                # Get ground truth from metadata
                if trace.metadata and isinstance(trace.metadata, dict):
                    trace_data["ground_truth_answer"] = trace.metadata.get(
                        "ground_truth", ""
                    )

                # Add scores from this trace
                if trace.scores:
                    for score in trace.scores:
                        score_name = score.name
                        score_value = score.value
                        score_comment = score.comment
                        # Get data_type from Langfuse score, default to NUMERIC
                        data_type = getattr(score, "data_type", None) or "NUMERIC"

                        # Build score entry for trace
                        # Round numeric values to 2 decimal places
                        if data_type != "CATEGORICAL" and isinstance(
                            score_value, (int, float)
                        ):
                            score_value = round(float(score_value), 2)

                        score_entry: TraceScore = {
                            "name": score_name,
                            "value": score_value,
                            "data_type": data_type,
                        }
                        if score_comment:
                            score_entry["comment"] = score_comment

                        trace_data["scores"].append(score_entry)

                        # Aggregate for summary calculation
                        if score_value is not None:
                            if score_name not in score_aggregations:
                                score_aggregations[score_name] = {
                                    "data_type": data_type,
                                    "values": [],
                                }
                            score_aggregations[score_name]["values"].append(score_value)

                traces.append(trace_data)

            except Exception as e:
                logger.warning(
                    f"[fetch_trace_scores_from_langfuse] Failed to fetch trace | "
                    f"trace_id={trace_id} | error={e}"
                )
                continue

        # 4. Identify complete scores (all traces must have the score)
        total_traces = len(traces)
        complete_score_names = {
            name
            for name, data in score_aggregations.items()
            if len(data["values"]) == total_traces
        }

        # 5. Filter trace scores to only include complete scores
        for trace in traces:
            trace["scores"] = [
                score
                for score in trace["scores"]
                if score["name"] in complete_score_names
            ]

        # 6. Calculate summary scores (only for complete scores)
        summary_scores = []
        for score_name, agg_data in score_aggregations.items():
            if score_name not in complete_score_names:
                continue

            data_type = agg_data["data_type"]
            values = agg_data["values"]

            if data_type == "CATEGORICAL":
                # For categorical scores, compute distribution
                distribution: dict[str, int] = {}
                for val in values:
                    str_val = str(val)
                    distribution[str_val] = distribution.get(str_val, 0) + 1

                summary_scores.append(
                    {
                        "name": score_name,
                        "distribution": distribution,
                        "total_pairs": len(values),
                        "data_type": data_type,
                    }
                )
            else:
                # For numeric scores, compute avg and std (rounded to 2 decimal places)
                numeric_values = [float(v) for v in values]
                summary_scores.append(
                    {
                        "name": score_name,
                        "avg": round(float(np.mean(numeric_values)), 2),
                        "std": round(float(np.std(numeric_values)), 2),
                        "total_pairs": len(numeric_values),
                        "data_type": data_type,
                    }
                )

        result: EvaluationScore = {
            "summary_scores": summary_scores,
            "traces": traces,
        }

        logger.info(
            f"[fetch_trace_scores_from_langfuse] Successfully fetched scores | "
            f"total_traces={len(traces)} | complete_scores={list(complete_score_names)}"
        )

        return result

    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"[fetch_trace_scores_from_langfuse] Failed to fetch trace scores | "
            f"dataset={dataset_name} | run={run_name} | {e}",
            exc_info=True,
        )
        raise
