"""TTS Evaluation CRUD operations."""

from .batch import start_tts_evaluation_batch
from .cron import poll_all_pending_tts_evaluations
from .dataset import (
    create_tts_dataset,
    get_tts_dataset_by_id,
    list_tts_datasets,
)
from .run import (
    create_tts_run,
    get_tts_run_by_id,
    list_tts_runs,
    update_tts_run,
)
from .result import (
    create_tts_results,
    get_tts_result_by_id,
    get_results_by_run_id,
    update_tts_human_feedback,
)

__all__ = [
    # Batch
    "start_tts_evaluation_batch",
    # Cron
    "poll_all_pending_tts_evaluations",
    # Dataset
    "create_tts_dataset",
    "get_tts_dataset_by_id",
    "list_tts_datasets",
    # Run
    "create_tts_run",
    "get_tts_run_by_id",
    "list_tts_runs",
    "update_tts_run",
    # Result
    "create_tts_results",
    "get_tts_result_by_id",
    "get_results_by_run_id",
    "update_tts_human_feedback",
]
