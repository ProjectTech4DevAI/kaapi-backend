"""TTS Evaluation CRUD operations."""

from .batch import TTSBatchSubmissionError, start_tts_evaluation_batch
from .cron import poll_all_pending_tts_evaluations
from .dataset import (
    create_tts_dataset,
    get_tts_dataset_by_id,
    list_tts_datasets,
)
from .result import (
    bulk_update_pending_tts_results,
    create_tts_results,
    get_results_by_run_id,
    get_tts_result_by_id,
    list_pending_tts_results_by_ids,
    update_tts_human_feedback,
)
from .run import (
    create_tts_run,
    finalize_tts_run_status,
    get_tts_run_by_id,
    list_tts_runs,
    update_tts_run,
)

__all__ = [
    # Batch
    "TTSBatchSubmissionError",
    "start_tts_evaluation_batch",
    # Cron
    "poll_all_pending_tts_evaluations",
    # Dataset
    "create_tts_dataset",
    "get_tts_dataset_by_id",
    "list_tts_datasets",
    # Run
    "create_tts_run",
    "finalize_tts_run_status",
    "get_tts_run_by_id",
    "list_tts_runs",
    "update_tts_run",
    # Result
    "bulk_update_pending_tts_results",
    "create_tts_results",
    "get_tts_result_by_id",
    "get_results_by_run_id",
    "list_pending_tts_results_by_ids",
    "update_tts_human_feedback",
]
