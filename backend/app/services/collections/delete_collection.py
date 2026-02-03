import logging
from uuid import UUID

from sqlmodel import Session
from asgi_correlation_id import correlation_id

from app.core.db import engine
from app.crud import CollectionCrud, CollectionJobCrud
from app.models import (
    CollectionJobStatus,
    CollectionJobUpdate,
    CollectionJob,
    CollectionJobPublic,
    CollectionIDPublic,
)
from app.models.collection import DeletionRequest
from app.services.collections.helpers import extract_error_message
from app.services.collections.providers.registry import get_llm_provider
from app.celery.utils import start_low_priority_job
from app.utils import send_callback, APIResponse


logger = logging.getLogger(__name__)


def start_job(
    db: Session,
    request: DeletionRequest,
    project_id: int,
    collection_job_id: UUID,
    organization_id: int,
) -> str:
    trace_id = correlation_id.get() or "N/A"

    job_crud = CollectionJobCrud(db, project_id)
    collection_job = job_crud.update(
        collection_job_id, CollectionJobUpdate(trace_id=trace_id)
    )

    task_id = start_low_priority_job(
        function_path="app.services.collections.delete_collection.execute_job",
        project_id=project_id,
        job_id=str(collection_job_id),
        collection_id=str(request.collection_id),
        trace_id=trace_id,
        request=request.model_dump(mode="json"),
        organization_id=organization_id,
    )

    logger.info(
        "[delete_collection.start_job] Job scheduled to delete collection | "
        f"Job_id={collection_job_id}, project_id={project_id}, task_id={task_id}, collection_id={request.collection_id}"
    )
    return collection_job_id


def build_success_payload(collection_job: CollectionJob, collection_id: UUID) -> dict:
    """
    success: true
    data: { job_id, status, collection: { id } }
    error: null
    metadata: null
    """
    collection_public = CollectionIDPublic(id=collection_id)
    job_public = CollectionJobPublic.model_validate(
        collection_job,
        update={"collection": collection_public},
    )
    return APIResponse.success_response(job_public).model_dump(
        mode="json",
        exclude_none=True,
    )


def build_failure_payload(
    collection_job: CollectionJob, collection_id: UUID, error_message: str
) -> dict:
    """
    success: false
    data: { job_id, status, collection: { id } }
    error: "something went wrong"
    metadata: null
    """
    collection_public = CollectionIDPublic(id=collection_id)
    job_public = CollectionJobPublic.model_validate(
        collection_job,
        update={"collection": collection_public},
    )
    return APIResponse.failure_response(
        extract_error_message(error_message), job_public
    ).model_dump(mode="json", exclude={"data": {"error_message"}})


def _mark_job_failed_and_callback(
    *,
    project_id: int,
    collection_id: UUID,
    job_id: UUID,
    err: Exception,
    callback_url: str | None,
) -> None:
    """
    Common failure handler:
    - mark job as FAILED with error_message
    - log error
    - send failure callback (if configured)
    """
    collection_job = None
    try:
        with Session(engine) as session:
            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job_crud.update(
                job_id,
                CollectionJobUpdate(
                    status=CollectionJobStatus.FAILED,
                    error_message=str(err),
                ),
            )
            collection_job = collection_job_crud.read_one(job_id)
    except Exception:
        logger.warning("[delete_collection.execute_job] Failed to mark job as FAILED")

    logger.error(
        "[delete_collection.execute_job] deletion failed | "
        "{'collection_id': '%s', 'error': '%s', 'job_id': '%s'}",
        str(collection_id),
        str(err),
        str(job_id),
        exc_info=True,
    )

    if callback_url and collection_job:
        failure_payload = build_failure_payload(
            collection_job=collection_job,
            collection_id=collection_id,
            error_message=str(err),
        )
        send_callback(callback_url, failure_payload)


def execute_job(
    request: dict,
    project_id: int,
    organization_id: int,
    task_id: str,
    job_id: str,
    collection_id: str,
    task_instance,
) -> None:
    """Celery worker entrypoint for deleting a collection (both remote and local)."""

    deletion_request = DeletionRequest(**request)

    collection_id = UUID(collection_id)
    job_uuid = UUID(job_id)

    collection_job = None

    try:
        with Session(engine) as session:
            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job = collection_job_crud.read_one(job_uuid)
            collection_job = collection_job_crud.update(
                job_uuid,
                CollectionJobUpdate(
                    task_id=task_id,
                    status=CollectionJobStatus.PROCESSING,
                ),
            )

            collection = CollectionCrud(session, project_id).read_one(collection_id)

            provider = get_llm_provider(
                session=session,
                provider=collection.provider,
                project_id=project_id,
                organization_id=organization_id,
            )

        provider.delete(collection)

        with Session(engine) as session:
            CollectionCrud(session, project_id).delete_by_id(collection_id)

            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job_crud.update(
                collection_job.id,
                CollectionJobUpdate(
                    status=CollectionJobStatus.SUCCESSFUL,
                    error_message=None,
                ),
            )
            collection_job = collection_job_crud.read_one(collection_job.id)

        logger.info(
            "[delete_collection.execute_job] Collection deleted successfully | "
            "{'collection_id': '%s', 'job_id': '%s'}",
            str(collection_id),
            str(job_uuid),
        )
        if deletion_request.callback_url and collection_job:
            success_payload = build_success_payload(
                collection_job=collection_job,
                collection_id=collection_id,
            )
            send_callback(deletion_request.callback_url, success_payload)

    except Exception as err:
        _mark_job_failed_and_callback(
            project_id=project_id,
            collection_id=collection_id,
            job_id=job_uuid,
            err=err,
            callback_url=deletion_request.callback_url,
        )
