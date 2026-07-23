# Module: Knowledge Base

Documents + collections for RAG: upload documents, transform them, group into provider vector-store collections.
Deep dive: `docs/architecture/kaapi-knowledge-base-ARCHITECTURE.md` (§3 upload, §5 collection lifecycle, §6 batching, §7 dedup, §8 provider abstraction, §10 deletion).

All paths relative to `backend/app/`.

## Routes
- `api/routes/documents.py` — upload/list
- `api/routes/collections.py`, `api/routes/collection_job.py` — collection CRUD + job status
- `api/routes/doc_transformation_job.py` — transform job status

## Tables (SQLModel)
| Table | Model |
|---|---|
| `document` (Document; self-FK parent→transformed child) | `models/document.py` |
| `collection` (Collection) | `models/collection.py` |
| `document_collection` (join) | `models/document_collection.py` |
| `collection_job` (CollectionJob) | `models/collection_job.py` |
| `doc_transformation_job` | `models/doc_transformation_job.py` |
| `file` (File) | `models/file.py` |

## Services / CRUD
- `services/collections/` — `create_collection.py`, `delete_collection.py`, `providers/`, `helpers.py`
- `services/documents/` — upload path
- `services/doctransform/` — `job.py`, `registry.py`, `transformer.py`, `zerox_transformer.py`
- `crud/collection/`, `crud/document/`, `crud/document_collection.py`, `crud/rag/`, `crud/file.py`

## Async
- Collection create/delete and doc transforms run as Celery jobs (`collection_job`, `doc_transformation_job` track state).

## External
- OpenAI vector stores / file uploads, object storage (`core/cloud/storage.py`), Zerox/OCR for transforms.

## Gotchas
- Uploads de-duplicate by provider file ID (see deep dive §7).
- Collections are immutable-ish: deletion semantics in deep dive §10.
- OpenAI file-batch id: the SDK's `file_batches.poll()` / `upload_and_poll()` final return deserializes a vector-store body, so its `.id` is the `vs_` id, not the `vsfb_` batch id. `crud/rag/open_ai.py` captures the batch id from `create()` before polling and uses it for `list_files`. Any failed file is a hard failure (whole vector store rolled back); partial indexing needs an add-documents endpoint first.
