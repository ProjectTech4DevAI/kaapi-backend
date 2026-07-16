"""v2 native LLM-as-judge run trigger.

Thin layer over the shared fast-eval pipeline: it marks the run as a judged
(Kaapi-native) run and reuses v1's `validate_and_start_fast_evaluation` for
dataset/config validation, run creation, and chunk dispatch. The judge itself
runs inside the aggregate, gated on the run's `is_judge_run` marker. Judging is
system-config only — always the fallback model + built-in prompt, no per-run
tailoring. v1's trigger is untouched.

See docs/srd-three-metric-evaluation-verdict.md for the full design.
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
    logger.info(
        f"[validate_and_start_judged_evaluation] Starting v2 judged eval | "
        f"run_name={run_name} | dataset_id={dataset_id} | "
        f"org_id={organization_id} | project_id={project_id}"
    )

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
