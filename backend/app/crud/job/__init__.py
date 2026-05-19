"""Job-related CRUD operations.

For batch operations (start_batch_job, poll_batch_status, etc.),
import directly from app.core.batch instead.
"""

from app.crud.job.job import (
    create_batch_job,
    delete_batch_job,
    get_batch_job,
    get_batch_jobs_by_ids,
    get_batches_by_type,
    update_batch_job,
)
from app.crud.job.pending_monitor import (
    count_stale_pending_collection_jobs,
    count_stale_pending_doc_transformation_jobs,
    count_stale_llm_pending_jobs,
)

__all__ = [
    # CRUD operations
    "create_batch_job",
    "get_batch_job",
    "update_batch_job",
    "get_batch_jobs_by_ids",
    "get_batches_by_type",
    "delete_batch_job",
    # Pending-job monitor reads
    "count_stale_llm_pending_jobs",
    "count_stale_pending_collection_jobs",
    "count_stale_pending_doc_transformation_jobs",
]
