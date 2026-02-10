"""STT Evaluation CRUD operations."""

from .batch import start_stt_evaluation_batch
from .cron import poll_all_pending_stt_evaluations
from .dataset import (
    create_stt_dataset,
    create_stt_samples,
    get_stt_dataset_by_id,
    list_stt_datasets,
    get_samples_by_dataset_id,
)
from .run import (
    create_stt_run,
    get_stt_run_by_id,
    list_stt_runs,
    update_stt_run,
)
from .result import (
    create_stt_results,
    get_stt_result_by_id,
    get_results_by_run_id,
    update_human_feedback,
)

__all__ = [
    # Batch
    "start_stt_evaluation_batch",
    # Cron
    "poll_all_pending_stt_evaluations",
    # Dataset
    "create_stt_dataset",
    "create_stt_samples",
    "get_stt_dataset_by_id",
    "list_stt_datasets",
    "get_samples_by_dataset_id",
    # Run
    "create_stt_run",
    "get_stt_run_by_id",
    "list_stt_runs",
    "update_stt_run",
    # Result
    "create_stt_results",
    "get_stt_result_by_id",
    "get_results_by_run_id",
    "update_human_feedback",
]
