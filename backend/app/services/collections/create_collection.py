import logging
import time
from uuid import UUID, uuid4

from sqlmodel import Session
from gevent import Timeout
from celery.exceptions import SoftTimeLimitExceeded
from opentelemetry import trace
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
    batch_documents,
    extract_error_message,
    to_collection_public,
)
from app.services.collections.providers.registry import get_llm_provider
from app.celery.utils import start_collection_setup_job, start_collection_batch_job
from app.utils import send_callback, APIResponse, get_webhook_secret


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def start_job(
    db: Session,
    request: CreationRequest,
    project_id: int,
    collection_job_id: UUID,
    organization_id: int,
) -> str:
    trace_id = correlation_id.get() or "N/A"

    job_crud = CollectionJobCrud(db, project_id)
    job_crud.update(collection_job_id, CollectionJobUpdate(trace_id=trace_id))

    task_id = start_collection_setup_job(
        project_id=project_id,
        job_id=str(collection_job_id),
        trace_id=trace_id,
        request=request.model_dump(mode="json"),
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
        logger.warning("[create_collection] Failed to mark job as FAILED")
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


def execute_setup_job(
    request: dict,
    project_id: int,
    organization_id: int,
    task_id: str,
    job_id: str,
    task_instance,
) -> None:
    """
    Phase 1: Fetch documents, create the vector store, split into batches,
    update job state to PROCESSING, then queue the first batch task.
    """
    collection_job = None
    creation_request = None

    with log_context(
        tag="collection",
        lifecycle="collection.create.execute_setup_job",
        action="create",
        collection_job_id=job_id,
        task_id=task_id,
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("collections.create.execute_setup_job") as span:
        span.set_attribute("collection.job_id", str(job_id))
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("kaapi.organization_id", organization_id)

        try:
            creation_request = CreationRequest(**request)
            span.set_attribute("collection.provider", str(creation_request.provider))

            job_uuid = UUID(job_id)
            trace_id = correlation_id.get() or "N/A"

            with Session(engine) as session:
                document_crud = DocumentCrud(session, project_id)
                flat_docs = document_crud.read_each(creation_request.documents)
                storage = get_cloud_storage(session=session, project_id=project_id)

                provider = get_llm_provider(
                    session=session,
                    provider=creation_request.provider,
                    project_id=project_id,
                    organization_id=organization_id,
                )

                for doc in flat_docs:
                    session.expunge(doc)

                collection_job_crud = CollectionJobCrud(session, project_id)
                collection_job = collection_job_crud.update(
                    job_uuid,
                    CollectionJobUpdate(
                        task_id=task_id,
                        status=CollectionJobStatus.PROCESSING,
                    ),
                )

            provider.upload_files(storage, flat_docs, project_id)

            logger.info(
                "[create_collection.execute_setup_job] All file uploads complete | "
                "job_id=%s, total=%d",
                job_id,
                len(flat_docs),
            )

            total_size_kb = sum(doc.file_size_kb for doc in flat_docs)
            total_size_mb = round(total_size_kb / 1024, 2)

            docs_batches = batch_documents(flat_docs)
            total_batches = len(docs_batches)
            batch_doc_ids = [[str(doc.id) for doc in batch] for batch in docs_batches]

            with Session(engine) as session:
                collection_job_crud = CollectionJobCrud(session, project_id)
                collection_job = collection_job_crud.update(
                    job_uuid,
                    CollectionJobUpdate(
                        total_size_mb=total_size_mb,
                        current_batch_number=0,
                        total_batches=total_batches,
                        documents_uploaded=[],
                    ),
                )

            start_collection_batch_job(
                project_id=project_id,
                job_id=job_id,
                trace_id=trace_id,
                batch_number=1,
                batch_doc_ids=batch_doc_ids[0],
                remaining_batches=batch_doc_ids[1:],
                request=request,
                vector_store_id=None,
                organization_id=organization_id,
            )

            logger.info(
                "[create_collection.execute_setup_job] Setup complete, first batch queued | "
                f"job_id={job_id}, total_batches={total_batches}"
            )

        except (Timeout, SoftTimeLimitExceeded) as err:
            timeout_err = TimeoutError("Task exceeded soft time limit")
            logger.warning(
                "[create_collection.execute_setup_job] Collection Creation Timed Out | {'collection_job_id': '%s', 'error': '%s'}",
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
            )
            raise

        except Exception as err:
            logger.error(
                "[create_collection.execute_setup_job] Setup failed | job_id=%s, error=%s",
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
            )
            raise


def execute_batch_job(
    request: dict,
    project_id: int,
    organization_id: int,
    task_id: str,
    job_id: str,
    task_instance,
    vector_store_id: str | None,
    batch_number: int,
    batch_doc_ids: list[str],
    remaining_batches: list[list[str]],
) -> None:
    """
    Phase 2: Upload one batch of documents to the vector store.
    - Uploads the batch via provider.create(); raises immediately on failure
    - Checkpoints progress to the DB
    - If more batches remain, queues the next batch task
    - If this is the last batch, finalizes: creates Collection, links docs, marks job SUCCESSFUL
    """
    collection_job = None
    result = None
    creation_request = None
    provider = None

    with log_context(
        tag="collection",
        lifecycle="collection.create.execute_batch_job",
        action="create",
        collection_job_id=job_id,
        task_id=task_id,
        project_id=project_id,
        organization_id=organization_id,
    ), tracer.start_as_current_span("collections.create.execute_batch_job") as span:
        span.set_attribute("collection.job_id", str(job_id))
        span.set_attribute("kaapi.project_id", project_id)
        span.set_attribute("kaapi.organization_id", organization_id)

        try:
            batch_start_time = time.time()
            creation_request = CreationRequest(**request)
            span.set_attribute("collection.provider", str(creation_request.provider))

            job_uuid = UUID(job_id)
            trace_id = correlation_id.get() or "N/A"

            logger.info(
                "[create_collection.execute_batch_job] Starting batch | "
                "job_id=%s, batch_number=%d, doc_count=%d, remaining_batches=%d",
                job_id,
                batch_number,
                len(batch_doc_ids),
                len(remaining_batches),
            )

            all_doc_ids_this_batch = [UUID(d) for d in batch_doc_ids]

            with Session(engine) as session:
                provider = get_llm_provider(
                    session=session,
                    provider=creation_request.provider,
                    project_id=project_id,
                    organization_id=organization_id,
                )

            with Session(engine) as session:
                document_crud = DocumentCrud(session, project_id)
                batch_docs = (
                    document_crud.read_each(all_doc_ids_this_batch)
                    if all_doc_ids_this_batch
                    else []
                )
                for doc in batch_docs:
                    session.expunge(doc)

            collection_result = provider.create(
                batch_docs,
                vector_store_id=vector_store_id,
            )
            result = collection_result
            resolved_vector_store_id = collection_result.llm_service_id

            with Session(engine) as session:
                collection_job_crud = CollectionJobCrud(session, project_id)
                collection_job = collection_job_crud.read_one(job_uuid)
                already_uploaded = collection_job.documents_uploaded or []
                now_uploaded = already_uploaded + [
                    str(d) for d in all_doc_ids_this_batch
                ]

                collection_job = collection_job_crud.update(
                    job_uuid,
                    CollectionJobUpdate(
                        current_batch_number=batch_number,
                        documents_uploaded=now_uploaded,
                    ),
                )

            logger.info(
                "[create_collection.execute_batch_job] Batch %d complete | "
                "doc_count=%d, job_id=%s",
                batch_number,
                len(all_doc_ids_this_batch),
                job_id,
            )

            if remaining_batches:
                start_collection_batch_job(
                    project_id=project_id,
                    job_id=job_id,
                    trace_id=trace_id,
                    vector_store_id=resolved_vector_store_id,
                    batch_number=batch_number + 1,
                    batch_doc_ids=remaining_batches[0],
                    remaining_batches=remaining_batches[1:],
                    request=request,
                    organization_id=organization_id,
                )
                logger.info(
                    "[create_collection.execute_batch_job] Batch %d/%d done, next batch queued | "
                    "job_id=%s, elapsed=%.2fs",
                    batch_number,
                    batch_number + len(remaining_batches),
                    job_id,
                    time.time() - batch_start_time,
                )
                return

            # Final batch: collection_result already has vector_store finalized
            finalize_start_time = time.time()

            with Session(engine) as session:
                all_uploaded_ids = [UUID(d) for d in now_uploaded]
                document_crud = DocumentCrud(session, project_id)
                all_docs = (
                    document_crud.read_each(all_uploaded_ids)
                    if all_uploaded_ids
                    else []
                )
                for doc in all_docs:
                    session.expunge(doc)

            with Session(engine) as session:
                collection_id = uuid4()
                collection = Collection(
                    id=collection_id,
                    project_id=project_id,
                    llm_service_id=collection_result.llm_service_id,
                    llm_service_name=collection_result.llm_service_name,
                    provider=creation_request.provider,
                    name=creation_request.name,
                    description=creation_request.description,
                )
                collection_crud = CollectionCrud(session, project_id)
                collection_crud.create(collection)
                collection = collection_crud.read_one(collection.id)

                if all_docs:
                    DocumentCollectionCrud(session).create(collection, all_docs)

                collection_job_crud = CollectionJobCrud(session, project_id)
                collection_job = collection_job_crud.update(
                    job_uuid,
                    CollectionJobUpdate(
                        status=CollectionJobStatus.SUCCESSFUL,
                        collection_id=collection.id,
                    ),
                )

                success_payload = build_success_payload(collection_job, collection)

            span.set_attribute("collection.id", str(collection_id))

            logger.info(
                "[create_collection.execute_batch_job] All batches done, collection created: %s | "
                "finalize_time=%.2fs, total_time=%.2fs, total_docs=%d",
                collection_id,
                time.time() - finalize_start_time,
                time.time() - batch_start_time,
                len(all_docs),
            )

            if creation_request.callback_url:
                webhook_secret = get_webhook_secret(project_id, organization_id)
                send_callback(
                    str(creation_request.callback_url),
                    success_payload,
                    webhook_secret=webhook_secret,
                )

        except (Timeout, SoftTimeLimitExceeded) as err:
            timeout_err = TimeoutError("Task exceeded soft time limit")
            logger.warning(
                "[create_collection.execute_batch_job] Collection Creation Timed Out | {'collection_job_id': '%s', 'error': '%s'}",
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
                "[create_collection.execute_batch_job] Collection Creation Failed | {'collection_job_id': '%s', 'error': '%s'}",
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
