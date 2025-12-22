"""Evaluation-related CRUD operations."""

from app.crud.evaluations.batch import start_evaluation_batch
from app.crud.evaluations.core import (
    create_evaluation_run,
    get_evaluation_run_by_id,
    list_evaluation_runs,
    resolve_model_from_config,
)
from app.crud.evaluations.cron import (
    process_all_pending_evaluations,
    process_all_pending_evaluations_sync,
)
from app.crud.evaluations.dataset import (
    create_evaluation_dataset,
    delete_dataset,
    get_dataset_by_id,
    list_datasets,
    upload_csv_to_object_store,
)
from app.crud.evaluations.embeddings import (
    calculate_average_similarity,
    calculate_cosine_similarity,
    start_embedding_batch,
)
from app.crud.evaluations.langfuse import (
    create_langfuse_dataset_run,
    update_traces_with_cosine_scores,
    upload_dataset_to_langfuse,
)
from app.crud.evaluations.processing import (
    check_and_process_evaluation,
    poll_all_pending_evaluations,
    process_completed_embedding_batch,
    process_completed_evaluation,
)

__all__ = [
    # Core
    "create_evaluation_run",
    "get_evaluation_run_by_id",
    "list_evaluation_runs",
    "resolve_model_from_config",
    # Cron
    "process_all_pending_evaluations",
    "process_all_pending_evaluations_sync",
    # Dataset
    "create_evaluation_dataset",
    "delete_dataset",
    "get_dataset_by_id",
    "list_datasets",
    "upload_csv_to_object_store",
    # Batch
    "start_evaluation_batch",
    # Processing
    "check_and_process_evaluation",
    "poll_all_pending_evaluations",
    "process_completed_embedding_batch",
    "process_completed_evaluation",
    # Embeddings
    "calculate_average_similarity",
    "calculate_cosine_similarity",
    "start_embedding_batch",
    # Langfuse
    "create_langfuse_dataset_run",
    "update_traces_with_cosine_scores",
    "upload_dataset_to_langfuse",
]
