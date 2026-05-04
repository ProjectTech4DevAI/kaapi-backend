import logging
import time
from uuid import UUID, uuid4

from opentelemetry import trace
from sqlmodel import Session
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout
from asgi_correlation_id import correlation_id

from app.core.cloud import get_cloud_storage
from app.core.db import engine
from app.core.telemetry import log_context
from app.crud import (
    CollectionCrud,
    DocumentCrud,
    DocumentCollectionCrud,
    CollectionJobCrud,
)
from app.models import (
    CollectionJobStatus,
    CollectionJob,
    Collection,
    CollectionJobUpdate,
    CollectionJobPublic,
    CreationRequest,
)
from app.services.collections.helpers import (
    extract_error_message,
    to_collection_public,
)
from app.services.collections.providers.registry import get_llm_provider
from app.celery.utils import start_create_collection_job
from app.utils import send_callback, get_webhook_secret, APIResponse


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def start_job(
    db: Session,
    request: CreationRequest,
    project_id: int,
    collection_job_id: UUID,
    with_assistant: bool,
    organization_id: int,
) -> str:
    with log_context(
        tag="collection",
        lifecycle="collection.create.start_job",
        action="create",
        collection_job_id=collection_job_id,
        project_id=project_id,
        organization_id=organization_id,
    ):
        trace_id = correlation_id.get() or "N/A"

        job_crud = CollectionJobCrud(db, project_id)
        collection_job = job_crud.update(
            collection_job_id, CollectionJobUpdate(trace_id=trace_id)
        )

        task_id = start_create_collection_job(
            project_id=project_id,
            job_id=str(collection_job_id),
            trace_id=trace_id,
            request=request.model_dump(mode="json"),
            with_assistant=with_assistant,
            organization_id=organization_id,
        )

        logger.info(
            "[create_collection.start_job] Job scheduled to create collection | "
            f"collection_job_id={collection_job_id}, project_id={project_id}, task_id={task_id}"
        )

        return collection_job_id


def build_success_payload(
    collection_job: CollectionJob, collection: Collection
) -> dict:
    """
    {
      "success": true,
      "data": { job fields + full collection },
      "error": null,
      "metadata": null
    }
    """
    collection_public = to_collection_public(collection)
    collection_dict = collection_public.model_dump(mode="json", exclude_none=True)

    job_public = CollectionJobPublic.model_validate(
        collection_job,
        update={"collection": collection_dict},
    )
    return APIResponse.success_response(job_public).model_dump(
        mode="json", exclude={"data": {"error_message"}}
    )


def build_failure_payload(collection_job: CollectionJob, error_message: str) -> dict:
    """
    {
      "success": false,
      "data": { job fields, collection: null },
      "error": "something went wrong",
      "metadata": null
    }
    """
    # ensure `collection` is explicitly null in the payload
    job_public = CollectionJobPublic.model_validate(
        collection_job,
        update={"collection": None},
    )
    return APIResponse.failure_response(
        extract_error_message(error_message), job_public
    ).model_dump(
        mode="json",
        exclude={"data": {"error_message"}},
    )


def _mark_job_failed(
    project_id: int,
    job_id: str,
    err: Exception,
    collection_job: CollectionJob | None,
) -> CollectionJob | None:
    """Update job row to FAILED with error_message; return latest job or None."""
    try:
        with Session(engine) as session:
            collection_job_crud = CollectionJobCrud(session, project_id)
            if collection_job is None:
                collection_job = collection_job_crud.read_one(UUID(job_id))
            collection_job = collection_job_crud.update(
                collection_job.id,
                CollectionJobUpdate(
                    status=CollectionJobStatus.FAILED,
                    error_message=str(err),
                ),
            )
            return collection_job
    except Exception:
        logger.warning("[create_collection.execute_job] Failed to mark job as FAILED")
        return None


def _handle_job_failure(
    span,
    project_id: int,
    organization_id: int,
    job_id: str,
    err: Exception,
    collection_job: CollectionJob | None,
    creation_request: CreationRequest | None,
    provider=None,
    result=None,
) -> None:
    """Record failure on span, clean up provider, mark job failed, and send failure callback."""
    span.record_exception(err)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(err)))

    if provider is not None and result is not None:
        try:
            provider.delete(result)
        except Exception:
            logger.warning("[create_collection.execute_job] Provider cleanup failed")

    collection_job = _mark_job_failed(
        project_id=project_id,
        job_id=job_id,
        err=err,
        collection_job=collection_job,
    )

    if creation_request and creation_request.callback_url and collection_job:
        failure_payload = build_failure_payload(collection_job, str(err))
        webhook_secret = get_webhook_secret(project_id, organization_id)
        send_callback(
            str(creation_request.callback_url),
            failure_payload,
            webhook_secret=webhook_secret,
        )


def execute_job(
    request: dict,
    with_assistant: bool,
    project_id: int,
    organization_id: int,
    task_id: str,
    job_id: str,
    task_instance,
) -> None:
    """
    Worker entrypoint scheduled by start_job.
    Orchestrates: job state, provider init, collection creation,
    optional assistant creation, collection persistence, linking, callbacks, and cleanup.
    """
    start_time = time.time()

    collection_job = None
    result = None
    creation_request = None
    provider = None

    with log_context(
        tag="collection",
        lifecycle="collection.create.execute_job",
        action="create",
        collection_job_id=job_id,
        task_id=task_id,
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("collections.create.execute_job") as span:
        span.set_attribute("collection.job_id", str(job_id))
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("kaapi.organization_id", organization_id)

        try:
            creation_request = CreationRequest(**request)
            if with_assistant:
                creation_request.provider = "openai"

            span.set_attribute("collection.provider", str(creation_request.provider))

            job_uuid = UUID(job_id)

            with Session(engine) as session:
                document_crud = DocumentCrud(session, project_id)
                flat_docs = document_crud.read_each(creation_request.documents)

            file_exts = {
                doc.fname.split(".")[-1] for doc in flat_docs if "." in doc.fname
            }
            total_size_kb = sum(doc.file_size_kb or 0 for doc in flat_docs)
            total_size_mb = round(total_size_kb / 1024, 2)
            span.set_attribute("collection.documents.count", len(flat_docs))
            span.set_attribute("collection.documents.total_size_mb", total_size_mb)

            with Session(engine) as session:
                collection_job_crud = CollectionJobCrud(session, project_id)
                collection_job = collection_job_crud.read_one(job_uuid)
                collection_job = collection_job_crud.update(
                    job_uuid,
                    CollectionJobUpdate(
                        task_id=task_id,
                        status=CollectionJobStatus.PROCESSING,
                        total_size_mb=total_size_mb,
                    ),
                )

                storage = get_cloud_storage(session=session, project_id=project_id)
                provider = get_llm_provider(
                    session=session,
                    provider=creation_request.provider,
                    project_id=project_id,
                    organization_id=organization_id,
                )

            with tracer.start_as_current_span("collections.create.provider"):
                result = provider.create(
                    collection_request=creation_request,
                    storage=storage,
                    documents=flat_docs,
                )

            llm_service_id = result.llm_service_id
            llm_service_name = result.llm_service_name

            with Session(engine) as session:
                collection_crud = CollectionCrud(session, project_id)
                collection_id = uuid4()

                collection = Collection(
                    id=collection_id,
                    project_id=project_id,
                    llm_service_id=llm_service_id,
                    llm_service_name=llm_service_name,
                    provider=creation_request.provider,
                    name=creation_request.name,
                    description=creation_request.description,
                )
                collection_crud.create(collection)
                collection = collection_crud.read_one(collection.id)

                if flat_docs:
                    DocumentCollectionCrud(session).create(collection, flat_docs)

                collection_job_crud = CollectionJobCrud(session, project_id)
                collection_job = collection_job_crud.update(
                    collection_job.id,
                    CollectionJobUpdate(
                        status=CollectionJobStatus.SUCCESSFUL,
                        collection_id=collection.id,
                    ),
                )

                success_payload = build_success_payload(collection_job, collection)

            span.set_attribute("collection.id", str(collection_id))

            elapsed = time.time() - start_time
            logger.info(
                "[create_collection.execute_job] Collection created: %s | Time: %.2fs | Files: %d | Total Size: %s MB | Types: %s",
                collection_id,
                elapsed,
                len(flat_docs),
                collection_job.total_size_mb,
                list(file_exts),
            )

            if creation_request.callback_url:
                webhook_secret = get_webhook_secret(project_id, organization_id)
                send_callback(
                    str(creation_request.callback_url),
                    success_payload,
                    webhook_secret=webhook_secret,
                )

        except (Timeout, SoftTimeLimitExceeded) as err:
            timeout_err = TimeoutError(f"Task exceeded soft time limit")
            logger.error(
                "[create_collection.execute_job] Collection Creation Timed Out | {'collection_job_id': '%s', 'error': '%s'}",
                job_id,
                str(timeout_err),
            )
            _handle_job_failure(
                span,
                project_id,
                organization_id,
                job_id,
                timeout_err,
                collection_job,
                creation_request,
                provider,
                result,
            )
            raise

        except Exception as err:
            logger.error(
                "[create_collection.execute_job] Collection Creation Failed | {'collection_job_id': '%s', 'error': '%s'}",
                job_id,
                str(err),
                exc_info=True,
            )
            _handle_job_failure(
                span,
                project_id,
                organization_id,
                job_id,
                err,
                collection_job,
                creation_request,
                provider,
                result,
            )
            raise
