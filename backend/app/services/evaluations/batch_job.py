import logging
from uuid import UUID

from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from sqlmodel import Session

from app.core.db import engine
from app.crud.evaluations import (
    get_evaluation_run_by_id,
    resolve_evaluation_config,
    start_evaluation_batch,
)
from app.crud.evaluations.core import update_evaluation_run
from app.models.evaluation import EvaluationRunUpdate
from app.utils import get_langfuse_client

logger = logging.getLogger(__name__)


def execute_evaluation_batch_submission(
    project_id: int,
    job_id: str,
    task_id: str,
    task_instance,
    organization_id: int,
    config_id: str,
    config_version: int,
    **kwargs,
) -> dict:
    run_id = int(job_id)
    logger.info(
        f"[execute_evaluation_batch_submission] Starting | run_id={run_id} | task={task_id}"
    )
    with Session(engine) as session:
        run = get_evaluation_run_by_id(
            session=session,
            evaluation_id=run_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if not run:
            return {"success": False, "error": "Run not found"}
        try:
            config, error = resolve_evaluation_config(
                session=session,
                config_id=UUID(str(config_id)),
                config_version=config_version,
                project_id=project_id,
            )
            if error:
                update_evaluation_run(
                    session=session,
                    eval_run=run,
                    update=EvaluationRunUpdate(status="failed", error_message=error),
                )
                return {"success": False, "error": error}

            langfuse = get_langfuse_client(
                session=session, org_id=organization_id, project_id=project_id
            )
            run = start_evaluation_batch(
                langfuse=langfuse,
                session=session,
                eval_run=run,
                params=config.completion.params,
                provider=config.completion.provider,
            )
            return {"success": True, "batch_job_id": run.batch_job_id}
        except (Timeout, SoftTimeLimitExceeded):
            logger.warning(
                f"[execute_evaluation_batch_submission] Timeout | run_id={run_id}"
            )
            update_evaluation_run(
                session=session,
                eval_run=run,
                update=EvaluationRunUpdate(
                    status="failed", error_message="Task exceeded soft time limit"
                ),
            )
            raise
        except Exception as e:
            logger.error(
                f"[execute_evaluation_batch_submission] Failed | run_id={run_id} | {e}",
                exc_info=True,
            )
            update_evaluation_run(
                session=session,
                eval_run=run,
                update=EvaluationRunUpdate(status="failed", error_message=str(e)),
            )
            raise
