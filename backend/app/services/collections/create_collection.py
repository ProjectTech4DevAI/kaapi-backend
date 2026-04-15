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
from app.services.collections.helpers import (
    batch_documents,
    extract_error_message,
    to_collection_public,
)
from app.services.collections.providers.registry import get_llm_provider
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

            total_size_kb = sum(doc.file_size_kb or 0 for doc in flat_docs)
            total_size_mb = round(total_size_kb / 1024, 2)
            docs_batches = batch_documents(flat_docs)
            total_batches = len(docs_batches)

            # Compute batch_doc_ids inside the session while docs are still attached
            batch_doc_ids = [[str(doc.id) for doc in batch] for batch in docs_batches]

            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job = collection_job_crud.update(
                job_uuid,
                CollectionJobUpdate(
                    task_id=task_id,
                    status=CollectionJobStatus.PROCESSING,
                    total_size_mb=total_size_mb,
                    total_batches=total_batches,
                    current_batch_number=0,
                    documents_uploaded=[],
                ),
            )

            provider = get_llm_provider(
                session=session,
                provider=creation_request.provider,
                project_id=project_id,
                organization_id=organization_id,
            )

        from app.crud.rag import OpenAIVectorStoreCrud

        vector_store = OpenAIVectorStoreCrud(provider.client).create()
        vector_store_id = vector_store.id

        start_collection_batch_job(
            project_id=project_id,
            job_id=job_id,
            trace_id=trace_id,
            batch_number=1,
            batch_doc_ids=batch_doc_ids[0],
            remaining_batches=batch_doc_ids[1:],
            failed_doc_ids=[],
            vector_store_id=vector_store_id,
            request=request,
            with_assistant=with_assistant,
            organization_id=organization_id,
        )

        logger.info(
            "[create_collection.execute_setup_job] Setup complete, first batch queued | "
            f"job_id={job_id}, vector_store_id={vector_store_id}, total_batches={total_batches}"
        )

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
    batch_number: int,
    batch_doc_ids: list[str],
    remaining_batches: list[list[str]],
    failed_doc_ids: list[str],
    vector_store_id: str,
) -> None:
    """
    Phase 2: Upload one batch of documents to the vector store.
    - Retries failed docs from the previous batch by prepending them
    - Checkpoints progress to the DB
    - If more batches remain, queues the next batch task
    - If this is the last batch, finalizes: creates Collection, links docs, marks job SUCCESSFUL
    """
    collection_job = None
    creation_request = None

    try:
        creation_request = CreationRequest(**request)
        if with_assistant:
            creation_request.provider = "openai"

        job_uuid = UUID(job_id)
        trace_id = correlation_id.get() or "N/A"

        logger.info(
            "[create_collection.execute_batch_job] Starting batch | "
            "job_id=%s, batch_number=%d, doc_count=%d, failed_from_prev=%d, remaining_batches=%d",
            job_id,
            batch_number,
            len(batch_doc_ids),
            len(failed_doc_ids),
            len(remaining_batches),
        )

        # Fetch documents for this batch (+ failed from previous batch prepended)
        all_doc_ids_this_batch = [UUID(d) for d in (failed_doc_ids + batch_doc_ids)]

        with Session(engine) as session:
            document_crud = DocumentCrud(session, project_id)
            docs_this_batch = document_crud.read_each(all_doc_ids_this_batch)
            storage = get_cloud_storage(session=session, project_id=project_id)

            provider = get_llm_provider(
                session=session,
                provider=creation_request.provider,
                project_id=project_id,
                organization_id=organization_id,
            )

        from app.crud.rag import OpenAIVectorStoreCrud

        vector_store_crud = OpenAIVectorStoreCrud(provider.client)
        logger.info(
            "[create_collection.execute_batch_job] Uploading batch to vector store | "
            "job_id=%s, batch_number=%d, vector_store_id=%s, total_docs=%d",
            job_id,
            batch_number,
            vector_store_id,
            len(all_doc_ids_this_batch),
        )
        succeeded, failed = vector_store_crud.update_batch(
            vector_store_id, storage, docs_this_batch
        )

        # Persist provider_file_ids and checkpoint progress in one session
        with Session(engine) as session:
            document_crud = DocumentCrud(session, project_id)
            for doc in succeeded:
                if doc.provider_file_id:
                    db_doc = document_crud.read_one(doc.id)
                    if db_doc.provider_file_id != doc.provider_file_id:
                        db_doc.provider_file_id = doc.provider_file_id
                        document_crud.update(db_doc)

            collection_job_crud = CollectionJobCrud(session, project_id)
            collection_job = collection_job_crud.read_one(job_uuid)
            already_uploaded = collection_job.documents_uploaded or []
            now_uploaded = already_uploaded + [str(doc.id) for doc in succeeded]

            collection_job = collection_job_crud.update(
                job_uuid,
                CollectionJobUpdate(
                    current_batch_number=batch_number,
                    documents_uploaded=now_uploaded,
                ),
            )

        logger.info(
            "[create_collection.execute_batch_job] Batch %d complete | "
            "succeeded=%d, failed=%d, job_id=%s",
            batch_number,
            len(succeeded),
            len(failed),
            job_id,
        )

        next_failed_doc_ids = [str(doc.id) for doc in failed]

        if remaining_batches:
            # Queue the next batch task
            start_collection_batch_job(
                project_id=project_id,
                job_id=job_id,
                trace_id=trace_id,
                batch_number=batch_number + 1,
                batch_doc_ids=remaining_batches[0],
                remaining_batches=remaining_batches[1:],
                failed_doc_ids=next_failed_doc_ids,
                vector_store_id=vector_store_id,
                request=request,
                with_assistant=with_assistant,
                organization_id=organization_id,
            )
            return

        # Last batch — do a final retry for any still-failing docs
        if next_failed_doc_ids:
            final_failed_docs = [
                doc for doc in docs_this_batch if str(doc.id) in next_failed_doc_ids
            ]
            final_succeeded, still_failed = vector_store_crud.update_batch(
                vector_store_id, storage, final_failed_docs
            )
            if still_failed:
                ids = [str(d.id) for d in still_failed]
                raise RuntimeError(
                    f"Failed to upload {len(still_failed)} document(s) after all retries: {ids}"
                )
            now_uploaded += [str(doc.id) for doc in final_succeeded]

        # Finalize: create Collection record, link all docs, mark job SUCCESSFUL
        start_time = time.time()

        with Session(engine) as session:
            all_uploaded_ids = [UUID(d) for d in now_uploaded]
            document_crud = DocumentCrud(session, project_id)
            all_docs = (
                document_crud.read_each(all_uploaded_ids) if all_uploaded_ids else []
            )

            with_assistant_flag = (
                creation_request.model is not None
                and creation_request.instructions is not None
            )

            if with_assistant_flag:
                assistant_crud_obj = __import__(
                    "app.crud.rag", fromlist=["OpenAIAssistantCrud"]
                ).OpenAIAssistantCrud(provider.client)
                assistant_options = {
                    k: v
                    for k, v in {
                        "model": creation_request.model,
                        "instructions": creation_request.instructions,
                        "temperature": creation_request.temperature,
                    }.items()
                    if v is not None
                }
                assistant = assistant_crud_obj.create(
                    vector_store_id, **assistant_options
                )
                llm_service_id = assistant.id
                llm_service_name = assistant_options.get("model", "assistant")
            else:
                from app.services.collections.helpers import get_service_name

                llm_service_id = vector_store_id
                llm_service_name = get_service_name("openai")

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

        elapsed = time.time() - start_time
        logger.info(
            "[create_collection.execute_batch_job] Collection created: %s | "
            "Time: %.2fs | Total docs: %d",
            collection_id,
            elapsed,
            len(all_docs),
        )

        if creation_request.callback_url:
            send_callback(creation_request.callback_url, success_payload)

    except Exception as err:
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
