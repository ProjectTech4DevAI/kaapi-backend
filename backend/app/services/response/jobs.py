import logging
from uuid import UUID
from fastapi import HTTPException
from sqlmodel import Session
from asgi_correlation_id import correlation_id
from app.crud import JobCrud
from app.models import JobType, JobStatus, JobUpdate, ResponsesAPIRequest
from app.celery.utils import start_response_job

from app.services.response.response import process_response
from app.services.response.callbacks import send_response_callback

logger = logging.getLogger(__name__)


def start_job(
    db: Session, request: ResponsesAPIRequest, project_id: int, organization_id: int
) -> UUID:
    """Create a response job and schedule Celery task."""
    trace_id = correlation_id.get() or "N/A"
    job_crud = JobCrud(session=db)
    job = job_crud.create(
        job_type=JobType.RESPONSE, trace_id=trace_id, project_id=project_id
    )

    try:
        task_id = start_response_job(
            project_id=project_id,
            job_id=str(job.id),
            trace_id=trace_id,
            request_data=request.model_dump(),
            organization_id=organization_id,
        )
    except Exception as e:
        logger.error(
            f"[start_job] Error starting Celery task : {str(e)} | job_id={job.id}, project_id={project_id}",
            exc_info=True,
        )
        job_update = JobUpdate(status=JobStatus.FAILED, error_message=str(e))
        job_crud.update(job_id=job.id, job_update=job_update)
        raise HTTPException(
            status_code=500, detail="Internal server error while generating response"
        )

    logger.info(
        f"[start_job] Job scheduled to generate response | job_id={job.id}, project_id={project_id}, task_id={task_id}"
    )
    return job.id


def execute_job(
    request_data: dict,
    project_id: int,
    organization_id: int,
    job_id: str,
    task_id: str,
    task_instance,
) -> None:
    """Celery task to process a response request asynchronously."""
    request_data: ResponsesAPIRequest = ResponsesAPIRequest(**request_data)
    job_id = UUID(job_id)

    response = process_response(
        request=request_data,
        project_id=project_id,
        organization_id=organization_id,
        job_id=job_id,
        task_id=task_id,
        task_instance=task_instance,
    )

    if request_data.callback_url:
        send_response_callback(
            callback_url=request_data.callback_url,
            callback_response=response,
            request_dict=request_data.model_dump(),
        )
