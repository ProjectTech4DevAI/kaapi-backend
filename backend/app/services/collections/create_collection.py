import logging
import time
from uuid import UUID, uuid4

from sqlmodel import Session
from asgi_correlation_id import correlation_id

from app.core.cloud import get_cloud_storage
from app.core.db import engine
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
from app.crud.rag import OpenAIVectorStoreCrud
from app.services.collections.helpers import (
    batch_documents,
    extract_error_message,
    to_collection_public,
)
from app.services.collections.providers.registry import get_llm_provider
from gevent import Timeout
from app.celery.utils import start_create_collection_job, start_collection_batch_job
from app.utils import send_callback, APIResponse


logger = logging.getLogger(__name__)


def start_job(
    db: Session,
    request: CreationRequest,
    project_id: int,
    collection_job_id: UUID,
    with_assistant: bool,
    organization_id: int,
) -> str:
    trace_id = correlation_id.get() or "N/A"

    job_crud = CollectionJobCrud(db, project_id)
    job_crud.update(collection_job_id, CollectionJobUpdate(trace_id=trace_id))

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


def _persist_succeeded_docs(succeeded: list, project_id: int) -> list[str]:
    with Session(engine) as session:
        document_crud = DocumentCrud(session, project_id)
        for doc in succeeded:
            if doc.openai_file_id:
                db_doc = document_crud.read_one(doc.id)
                if db_doc.openai_file_id != doc.openai_file_id:
                    db_doc.openai_file_id = doc.openai_file_id
                    document_crud.update(db_doc)
    return [str(doc.id) for doc in succeeded]


def _retry_failed_uploads(
    vector_store_crud,
    vector_store_id: str,
    failed_docs: list,
    project_id: int,
    max_retries: int = 3,
) -> list[str]:
    """
    Retry attaching docs that failed the initial batch upload_and_poll.
    All docs must already have provider_file_id set.
    Returns the list of successfully retried doc IDs.
    Raises RuntimeError if any docs still fail after all retries.
    """
    pending = failed_docs
    all_succeeded_ids: list[str] = []

    for attempt in range(1, max_retries + 1):
        logger.warning(
            "[_retry_failed_uploads] Retry attempt %d/%d: %d doc(s) | vector_store_id=%s",
            attempt,
            max_retries,
            len(pending),
            vector_store_id,
        )
        succeeded, failed = vector_store_crud.update_batch(vector_store_id, pending)

        if succeeded:
            all_succeeded_ids += _persist_succeeded_docs(succeeded, project_id)

        if not failed:
            return all_succeeded_ids

        pending = failed

    ids = [str(d.id) for d in pending]
    raise RuntimeError(
        f"Failed to upload {len(pending)} document(s) after {max_retries} retries: {ids}"
    )


def execute_setup_job(
    request: dict,
    with_assistant: bool,
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

    try:
        creation_request = CreationRequest(**request)
        if with_assistant:
            creation_request.provider = "openai"

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

        provider.upload_files(storage, flat_docs, project_id)

        logger.info(
            "[create_collection.execute_setup_job] All file uploads complete | "
            "job_id=%s, total=%d, failed=%d, duration_s=%.2f",
            job_id,
            len(flat_docs),
        )

        total_size_kb = sum(doc.file_size_kb for doc in flat_docs)
        total_size_mb = total_size_kb / 1024

        docs_batches = batch_documents(flat_docs)
        total_batches = len(docs_batches)
        batch_doc_ids = [[str(doc.id) for doc in batch] for batch in docs_batches]

        with Session(engine) as session:
            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job = collection_job_crud.update(
                job_uuid,
                CollectionJobUpdate(
                    task_id=task_id,
                    status=CollectionJobStatus.PROCESSING,
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
            with_assistant=with_assistant,
            organization_id=organization_id,
        )

        logger.info(
            "[create_collection.execute_setup_job] Setup complete, first batch queued | "
            f"job_id={job_id}, total_batches={total_batches}"
        )

    except Timeout as err:
        timeout_err = TimeoutError(
            f"[execute_setup_job] Task exceeded soft time limit of {err.seconds}s"
        )
        _mark_job_failed(
            project_id=project_id,
            job_id=job_id,
            err=timeout_err,
            collection_job=collection_job,
        )
        raise

    except Exception as err:
        logger.error(
            "[create_collection.execute_setup_job] Setup failed | job_id=%s, error=%s",
            job_id,
            str(err),
            exc_info=True,
        )

        collection_job = _mark_job_failed(
            project_id=project_id,
            job_id=job_id,
            err=err,
            collection_job=collection_job,
        )
        if creation_request and creation_request.callback_url and collection_job:
            failure_payload = build_failure_payload(collection_job, str(err))
            send_callback(creation_request.callback_url, failure_payload)


def execute_batch_job(
    request: dict,
    with_assistant: bool,
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
    - Uploads the batch; any failures within the batch are retried inline by _upload_batch_with_retry
    - Raises immediately if all retries for the batch are exhausted
    - Checkpoints progress to the DB
    - If more batches remain, queues the next batch task
    - If this is the last batch, finalizes: creates Collection, links docs, marks job SUCCESSFUL
    """
    collection_job = None
    creation_request = None

    try:
        batch_start_time = time.time()
        creation_request = CreationRequest(**request)
        if with_assistant:
            creation_request.provider = "openai"

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
        is_final = not remaining_batches

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
            creation_request,
            batch_docs,
            vector_store_id=vector_store_id,
            is_final=is_final,
        )
        resolved_vector_store_id = (
            collection_result.llm_service_id
            if not is_final
            else vector_store_id or collection_result.llm_service_id
        )

        with Session(engine) as session:
            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job = collection_job_crud.read_one(job_uuid)
            already_uploaded = collection_job.documents_uploaded or []
            now_uploaded = already_uploaded + [str(d) for d in all_doc_ids_this_batch]

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
                with_assistant=with_assistant,
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

        # Final batch: collection_result already has assistant/vector_store finalized
        finalize_start_time = time.time()

        with Session(engine) as session:
            all_uploaded_ids = [UUID(d) for d in now_uploaded]
            document_crud = DocumentCrud(session, project_id)
            all_docs = (
                document_crud.read_each(all_uploaded_ids) if all_uploaded_ids else []
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

        logger.info(
            "[create_collection.execute_batch_job] All batches done, collection created: %s | "
            "finalize_time=%.2fs, total_time=%.2fs, total_docs=%d",
            collection_id,
            time.time() - finalize_start_time,
            time.time() - batch_start_time,
            len(all_docs),
        )

        if creation_request.callback_url:
            send_callback(creation_request.callback_url, success_payload)

    except Timeout as err:
        timeout_err = TimeoutError(
            f"[execute_batch_job] Task exceeded soft time limit of {err.seconds}s"
        )
        _mark_job_failed(
            project_id=project_id,
            job_id=job_id,
            err=timeout_err,
            collection_job=collection_job,
        )
        raise
    except BaseException as err:
        logger.error(
            "[create_collection.execute_batch_job] Batch %d failed | job_id=%s, error=%s",
            batch_number,
            job_id,
            str(err),
            exc_info=True,
        )
        collection_job = _mark_job_failed(
            project_id=project_id,
            job_id=job_id,
            err=err,
            collection_job=collection_job,
        )
        if creation_request and creation_request.callback_url and collection_job:
            failure_payload = build_failure_payload(collection_job, str(err))
            send_callback(creation_request.callback_url, failure_payload)
