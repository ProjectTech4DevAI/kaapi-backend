from typing import Optional, Tuple, Iterable, Union
from uuid import UUID

from fastapi import HTTPException

from app.services.doctransform.registry import (
    get_available_transformers,
    get_file_format,
    is_transformation_supported,
    resolve_transformer,
)
from app.crud import DocTransformationJobCrud, DocumentCrud
from app.services.doctransform import job as transformation_job
from app.models import (
    DocTransformJobCreate,
    TransformationStatus,
    TransformationJobInfo,
    Document,
    DocumentPublic,
    DocTransformationJob,
    DocTransformationJobPublic,
    TransformedDocumentPublic,
)


def pre_transform_validation(
    *,
    src_filename: str,
    target_format: str | None,
    transformer: str | None,
) -> Tuple[str, str | None]:
    """
    Everything BEFORE storage:
      - detect source_format
      - validate (source -> target) support if target requested
      - resolve actual transformer (or None if no target_format)

    Returns: (source_format, actual_transformer_or_none)
    Raises: HTTPException(400) on client errors.
    """
    try:
        source_format = get_file_format(src_filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    actual_transformer: Optional[str] = None
    if target_format:
        if not is_transformation_supported(source_format, target_format):
            raise HTTPException(
                status_code=400,
                detail=f"Transformation from {source_format} to {target_format} is not supported",
            )

        candidate = transformer or "default"
        try:
            actual_transformer = resolve_transformer(
                source_format, target_format, candidate
            )
        except ValueError as e:
            available = get_available_transformers(source_format, target_format)
            raise HTTPException(
                status_code=400,
                detail=f"{e}. Available transformers: {list(available.keys())}",
            )

    return source_format, actual_transformer


def schedule_transformation(
    *,
    session,
    project_id: int,
    source_format: str,
    target_format: str | None,
    actual_transformer: str | None,
    source_document_id: UUID,
    callback_url: str | None,
) -> TransformationJobInfo | None:
    """
    Everything AFTER the document row is persisted:
      - if target was requested and a transformer was resolved,
        create the job and enqueue it; return job_info.
      - otherwise return None.
    """
    if not (target_format and actual_transformer):
        return None

    job_crud = DocTransformationJobCrud(session, project_id)
    job = job_crud.create(DocTransformJobCreate(source_document_id=source_document_id))

    transformation_job_id = transformation_job.start_job(
        db=session,
        job_id=job.id,
        project_id=project_id,
        transformer_name=actual_transformer,
        target_format=target_format,
        callback_url=callback_url,
    )

    return TransformationJobInfo(
        message=f"Document accepted for transformation from {source_format} to {target_format}.",
        job_id=str(transformation_job_id),
        status=TransformationStatus.PENDING,
        transformer=actual_transformer,
        status_check_url=f"/documents/transformation/{transformation_job_id}",
    )


PublicDoc = Union[DocumentPublic, TransformedDocumentPublic]


def _to_public_schema(doc: Document) -> PublicDoc:
    if doc.source_document_id is None:
        return DocumentPublic.model_validate(doc, from_attributes=True)
    return TransformedDocumentPublic.model_validate(doc, from_attributes=True)


def build_document_schema(
    *,
    document: Document,
    include_url: bool,
    storage: object | None,
) -> PublicDoc:
    schema = _to_public_schema(document)
    if include_url and storage:
        schema.signed_url = storage.get_signed_url(document.object_store_url)
    return schema


def build_document_schemas(
    *,
    documents: Iterable[Document],
    include_url: bool,
    storage: object | None,
) -> list[PublicDoc]:
    out: list[PublicDoc] = []
    for doc in documents:
        schema = _to_public_schema(doc)
        if include_url and storage:
            schema.signed_url = storage.get_signed_url(doc.object_store_url)
        out.append(schema)
    return out


def build_job_schema(
    *,
    job: DocTransformationJob,
    doc_crud: DocumentCrud,
    include_url: bool,
    storage: object | None,
) -> DocTransformationJobPublic:
    """Build a single job schema, optionally attaching a signed URL."""
    transformed_doc_schema: TransformedDocumentPublic | None = None
    object_url: str | None = None

    if job.transformed_document_id:
        doc = doc_crud.read_one(job.transformed_document_id)
        transformed_doc_schema = TransformedDocumentPublic.model_validate(
            doc, from_attributes=True
        )
        object_url = doc.object_store_url

        if include_url and storage and object_url:
            transformed_doc_schema.signed_url = storage.get_signed_url(object_url)

    job_schema = DocTransformationJobPublic.model_validate(job, from_attributes=True)
    return job_schema.model_copy(
        update={"transformed_document": transformed_doc_schema}
    )


def build_job_schemas(
    *,
    jobs: Iterable[DocTransformationJob],
    doc_crud: DocumentCrud,
    include_url: bool,
    storage: object | None,
) -> list[DocTransformationJobPublic]:
    """Build many job schemas efficiently."""
    out: list[DocTransformationJobPublic] = []
    for job in jobs:
        out.append(
            build_job_schema(
                job=job,
                doc_crud=doc_crud,
                include_url=include_url,
                storage=storage,
            )
        )
    return out
