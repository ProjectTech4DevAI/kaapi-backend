"""Assessment-related CRUD operations."""

# New API-client crud is namespaced under `api` (its `create_assessment` /
# `get_assessment` would otherwise shadow the legacy core functions of the same
# name). Callers: `from app.crud.assessment import api; api.create_assessment(...)`.
from app.crud.assessment import api
from app.crud.assessment.core import (
    build_run_stats,
    compute_run_counts,
    create_assessment,
    create_assessment_run,
    derive_aggregate_error,
    derive_assessment_status,
    get_assessment_by_id,
    get_assessment_run_by_id,
    get_assessment_runs_for_assessment,
    list_assessment_runs,
    list_assessments,
    recompute_assessment_status,
    update_assessment_run_prefilter_stats,
    update_assessment_run_status,
    update_run_post_processing_config,
)
from app.crud.assessment.dataset import (
    create_assessment_dataset,
    delete_assessment_dataset,
    get_assessment_dataset_by_id,
    list_assessment_datasets,
)
from app.models.assessment import AssessmentRunCounts, AssessmentRunStat

__all__ = [
    "AssessmentRunCounts",
    "AssessmentRunStat",
    "api",
    "build_run_stats",
    "compute_run_counts",
    "create_assessment_dataset",
    "create_assessment",
    "create_assessment_run",
    "delete_assessment_dataset",
    "derive_aggregate_error",
    "derive_assessment_status",
    "get_assessment_by_id",
    "get_assessment_dataset_by_id",
    "get_assessment_run_by_id",
    "get_assessment_runs_for_assessment",
    "list_assessment_runs",
    "list_assessment_datasets",
    "list_assessments",
    "recompute_assessment_status",
    "update_assessment_run_prefilter_stats",
    "update_assessment_run_status",
    "update_run_post_processing_config",
]
