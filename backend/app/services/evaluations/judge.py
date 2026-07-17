"""Starts a v2 native LLM-as-judge run using the shared fast-eval pipeline.
Judging runs in the aggregate with the built-in prompt and fallback model; v1 remains unchanged.
"""

import logging
from uuid import UUID

from sqlmodel import Session

from app.models.evaluation import EvaluationRun
from app.services.evaluations.fast import validate_and_start_fast_evaluation

logger = logging.getLogger(__name__)


def validate_and_start_judged_evaluation(
    *,
    session: Session,
    dataset_id: int,
    run_name: str,
    config_id: UUID,
    config_version: int,
    organization_id: int,
    project_id: int,
    trace_id: str = "N/A",
) -> EvaluationRun:
    """Start a v2 judged fast evaluation run.

    Delegates dataset/config validation, run creation, and chunk dispatch to the
    shared v1 fast trigger with the native-judge marker set. Judging always runs
    for v2 fast runs — there is no opt-in flag and no per-run judge config.
    """
    return validate_and_start_fast_evaluation(
        session=session,
        dataset_id=dataset_id,
        run_name=run_name,
        config_id=config_id,
        config_version=config_version,
        organization_id=organization_id,
        project_id=project_id,
        trace_id=trace_id,
        is_judge_run=True,
    )
