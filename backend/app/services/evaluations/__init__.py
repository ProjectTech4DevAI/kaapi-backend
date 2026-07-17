"""Evaluation services."""

from app.services.evaluations.dataset import upload_dataset, upload_dataset_v2
from app.services.evaluations.evaluation import (
    get_evaluation_with_scores,
    validate_and_start_batch_evaluation,
)
from app.services.evaluations.fast import (
    execute_fast_evaluation_aggregate,
    execute_fast_evaluation_chunk,
    is_dataset_fast_eligible,
    validate_and_start_fast_evaluation,
)
from app.services.evaluations.prompt_improvement import (
    execute_prompt_improvement,
    start_prompt_improvement_job,
    validate_improve_prompt,
)
from app.services.evaluations.validators import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE,
    parse_csv_items,
    sanitize_dataset_name,
    validate_csv_file,
)

__all__ = [
    "ALLOWED_EXTENSIONS",
    "ALLOWED_MIME_TYPES",
    "MAX_FILE_SIZE",
    "execute_fast_evaluation_aggregate",
    "execute_fast_evaluation_chunk",
    "get_evaluation_with_scores",
    "improve_prompt",
    "is_dataset_fast_eligible",
    "parse_csv_items",
    "sanitize_dataset_name",
    "upload_dataset",
    "upload_dataset_v2",
    "validate_and_start_batch_evaluation",
    "validate_and_start_fast_evaluation",
    "validate_csv_file",
]
