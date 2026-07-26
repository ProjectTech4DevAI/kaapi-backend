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
- The SDK's `file_batches.poll()` never times out. `_poll_file_batch` polls `retrieve` itself under a fixed `BATCH_POLL_TIMEOUT_SECONDS` deadline. It used to take that deadline from the caller via a `task_budget` contextvar; that was deleted once the batch retry landed, because overrunning the soft limit stopped being fatal. Don't reintroduce caller coupling here.
- Retries are the SDK's own (`max_retries=OPENAI_MAX_RETRIES`, `timeout=OPENAI_TIMEOUT_SECONDS` in `providers/registry.py`). It covers connection errors/408/409/429/5xx, honours `retry-after`, and rewinds the upload stream between attempts (verified). Do not add a second retry layer: stacking gives 3x3=9 requests per call, and a custom tenacity layer measured 4.5x slower than the SDK on an intermittent-failure batch.
- `OPENAI_TIMEOUT_SECONDS` (90s) is a **stall detector, not an upload deadline**. httpx has no total-request timeout — connect/read/write/pool are each per-socket-operation — so a big document that streams steadily uploads fine however long it takes; the timeout only fires after 90 consecutive seconds of zero bytes. Verified: a 10MB body taking 12s under a 2s write timeout succeeds, and only a stalled receiver raises `WriteTimeout`. Don't raise it out of fear that large files get cut off — they don't. Its actual job is bounding how long we sit on a *dead* socket (SDK default is 600s, so 3 attempts = 1800s of nothing).
- A batch that hits the Celery soft limit is **retried** (`task_instance.retry()`, max 3), not failed. The retry skips `_handle_job_failure` so the vector store survives and is reused. This converges because `upload_files` persists each doc's file ID as it uploads, so each attempt does strictly less work — measured: 200 docs against a provider managing ~70 uploads/attempt finishes on attempt 3. Setup does not retry (a second `create_vector_store()` orphans the first). Note `celery.exceptions.Retry` subclasses `Exception`; it escapes only because it is raised from inside an `except` clause, which sibling `except Exception` handlers do not catch.
- `documents_uploaded` is deduped on append — a batch retried past its checkpoint re-adds its own IDs, and `DocumentCrud.read_each` raises when duplicates collapse in its `IN` clause.
- Do not "fix" batch timing with a self-requeueing continuation task — a task that returns normally is not redelivered, and a lost continuation strands the job in `PROCESSING`, which nothing monitors or recovers. Deep dive §11.5.
- Uploads happen per batch in `execute_batch_job`, not in setup; setup only plans batches and creates the vector store.
