"""STT Evaluation CRUD operations."""

from .dataset import (
    create_stt_dataset,
    create_stt_samples,
    get_stt_dataset_by_id,
    list_stt_datasets,
    get_samples_by_dataset_id,
    get_sample_count_for_dataset,
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
    update_stt_result,
    update_human_feedback,
    count_results_by_status,
)

__all__ = [
    # Dataset
    "create_stt_dataset",
    "create_stt_samples",
    "get_stt_dataset_by_id",
    "list_stt_datasets",
    "get_samples_by_dataset_id",
    "get_sample_count_for_dataset",
    # Run
    "create_stt_run",
    "get_stt_run_by_id",
    "list_stt_runs",
    "update_stt_run",
    # Result
    "create_stt_results",
    "get_stt_result_by_id",
    "get_results_by_run_id",
    "update_stt_result",
    "update_human_feedback",
    "count_results_by_status",
]
