import tempfile
import shutil
import time
import logging
from pathlib import Path
from uuid import uuid4, UUID

from fastapi import UploadFile
from tenacity import retry, wait_exponential, stop_after_attempt
from sqlmodel import Session
from asgi_correlation_id import correlation_id
from starlette.datastructures import Headers

from app.crud.document.doc_transformation_job import DocTransformationJobCrud
from app.crud.document.document import DocumentCrud
from app.models import (
    Document,
    DocTransformJobUpdate,
    TransformationStatus,
    DocTransformationJobPublic,
    TransformedDocumentPublic,
    DocTransformationJob,
)
from app.core.cloud import get_cloud_storage
from app.celery.utils import start_low_priority_job
from app.utils import send_callback, APIResponse
from app.services.doctransform.registry import convert_document, FORMAT_TO_EXTENSION
from app.core.db import engine

logger = logging.getLogger(__name__)


def start_job(
    db: Session,
    project_id: int,
    job_id: UUID,
    transformer_name: str,
    target_format: str,
    callback_url: str | None,
) -> str:
    trace_id = correlation_id.get() or "N/A"
    job_crud = DocTransformationJobCrud(db, project_id=project_id)
    job_crud.update(job_id, DocTransformJobUpdate(trace_id=trace_id))
    job = job_crud.read_one(job_id)

    task_id = start_low_priority_job(
        function_path="app.services.doctransform.job.execute_job",
        project_id=project_id,
        job_id=str(job.id),
        source_document_id=str(job.source_document_id),
        trace_id=trace_id,
        transformer_name=transformer_name,
        target_format=target_format,
        callback_url=callback_url,
    )

    logger.info(
        f"[start_job] Job scheduled for document transformation | id: {job.id}, project_id: {project_id}, task_id: {task_id}"
    )
    return job.id


def build_success_payload(
    job: DocTransformationJob,
    transformed_doc: TransformedDocumentPublic,
):
    """
    {
      "success": true,
      "data": { job fields + transformed_document (full) },
      "error": null,
      "metadata": null
    }
    """
    transformed_public = TransformedDocumentPublic.model_validate(transformed_doc)
    job_public = DocTransformationJobPublic.model_validate(
        job,
        update={"transformed_document": transformed_public},
    )
    # keep error_message out of the data envelope
    return APIResponse.success_response(job_public).model_dump(
        mode="json", exclude={"data": {"error_message"}}
    )


def build_failure_payload(job: DocTransformationJob, error_message: str):
    """
    {
      "success": false,
      "data": { job fields, transformed_document: null },
      "error": "something went wrong",
      "metadata": null
    }
    """
    # ensure transformed_document is explicitly null in the payload
    job_public = DocTransformationJobPublic.model_validate(
        job,
        update={"transformed_document": None},
    )
    return APIResponse.failure_response(error_message, job_public).model_dump(
        mode="json",
        exclude={"data": {"error_message"}},
    )


@retry(wait=wait_exponential(multiplier=5, min=5, max=10), stop=stop_after_attempt(3))
def execute_job(
    project_id: int,
    job_id: str,
    source_document_id: str,
    transformer_name: str,
    target_format: str,
    task_id: str,
    callback_url: str | None,
    task_instance,
):
    start_time = time.time()
    tmp_dir: Path | None = None

    job_for_payload = None  # keep latest job snapshot for payloads

    try:
        job_uuid = UUID(job_id)
        source_uuid = UUID(source_document_id)

        logger.info(
            "[doc_transform.execute_job] started | job_id=%s | transformer=%s | target=%s | project_id=%s",
            job_uuid,
            transformer_name,
            target_format,
            project_id,
        )

        with Session(engine) as db:
            job_crud = DocTransformationJobCrud(session=db, project_id=project_id)
            job_for_payload = job_crud.update(
                job_uuid,
                DocTransformJobUpdate(
                    status=TransformationStatus.PROCESSING, task_id=task_id
                ),
            )

            doc_crud = DocumentCrud(session=db, project_id=project_id)
            source_doc = doc_crud.read_one(source_uuid)

            source_doc_id = source_doc.id
            source_doc_fname = source_doc.fname
            source_doc_object_store_url = source_doc.object_store_url

            storage = get_cloud_storage(session=db, project_id=project_id)

        # --- download source ---
        body = storage.stream(source_doc_object_store_url)
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_in = tmp_dir / f"{source_doc_id}"
        with open(tmp_in, "wb") as f:
            shutil.copyfileobj(body, f)

        # --- transform ---
        fname_no_ext = Path(source_doc_fname).stem
        target_extension = FORMAT_TO_EXTENSION.get(target_format, f".{target_format}")
        transformed_doc_id = uuid4()
        tmp_out = tmp_dir / f"<transformed>{fname_no_ext}{target_extension}"

        convert_document(tmp_in, tmp_out, transformer_name)

        # --- upload transformed file ---
        content_type_map = {"markdown": "text/markdown; charset=utf-8"}
        content_type = content_type_map.get(target_format, "text/plain")

        with open(tmp_out, "rb") as fobj:
            file_upload = UploadFile(
                filename=tmp_out.name,
                file=fobj,
                headers=Headers({"content-type": content_type}),
            )
            dest = storage.put(file_upload, Path(str(transformed_doc_id)))

        with Session(engine) as db:
            new_doc = Document(
                id=transformed_doc_id,
                project_id=project_id,
                fname=tmp_out.name,
                object_store_url=str(dest),
                source_document_id=source_doc_id,
            )
            created = DocumentCrud(db, project_id).update(new_doc)

            job_crud = DocTransformationJobCrud(session=db, project_id=project_id)
            job_for_payload = job_crud.update(
                job_uuid,
                DocTransformJobUpdate(
                    status=TransformationStatus.COMPLETED,
                    transformed_document_id=created.id,
                ),
            )

            signed_url = None
            try:
                get_signed_url = getattr(storage, "get_signed_url", None)
                if callable(get_signed_url):
                    signed_url = get_signed_url(created.object_store_url)
            except Exception as e:
                logger.warning(
                    "[doc_transform] failed to generate signed URL for doc %s: %s",
                    created.id,
                    e,
                )

            transformed_public = TransformedDocumentPublic.model_validate(
                created,
                update={"signed_url": signed_url} if signed_url else None,
            )

            success_payload = build_success_payload(job_for_payload, transformed_public)

        elapsed = time.time() - start_time
        logger.info(
            "[doc_transform.execute_job] completed | job_id=%s | transformed_doc_id=%s | time=%.2fs",
            job_uuid,
            created.id,
            elapsed,
        )

        if callback_url:
            send_callback(callback_url, success_payload)

    except Exception as e:
        logger.error(
            "[doc_transform.execute_job] FAILED | job_id=%s | error=%s",
            job_uuid,
            e,
            exc_info=True,
        )

        try:
            with Session(engine) as db:
                job_crud = DocTransformationJobCrud(session=db, project_id=project_id)
                job_for_payload = job_crud.update(
                    job_uuid,
                    DocTransformJobUpdate(
                        status=TransformationStatus.FAILED, error_message=str(e)
                    ),
                )
        except Exception as db_error:
            logger.error(
                "[doc_transform.execute_job] failed to persist FAILED status | job_id=%s | db_error=%s",
                job_uuid,
                db_error,
            )

        if callback_url and job_for_payload:
            try:
                failure_payload = build_failure_payload(job_for_payload, str(e))
                send_callback(callback_url, failure_payload)
            except Exception as cb_error:
                logger.error(
                    "[doc_transform.execute_job] callback failed | job_id=%s | error=%s",
                    job_uuid,
                    cb_error,
                )

        # bubble up for caller/infra
        raise
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
