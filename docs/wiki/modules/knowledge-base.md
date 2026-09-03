# Module: Knowledge Base

Documents + collections for RAG: upload documents, transform them, group into provider vector-store collections.
Deep dive: `docs/architecture/kaapi-knowledge-base-ARCHITECTURE.md` (§3 upload, §5 collection lifecycle, §6 batching, §7 dedup, §8 provider abstraction, §10 deletion).

All paths relative to `backend/app/`.

## Routes
- `api/routes/documents.py` — upload/list (v1 multipart upload, the only path that transforms)
- `api/routes/documents_v2.py` — v2 pre-signed upload: `POST /upload-url` (issues a PUT URL) then `POST ""` (registers the uploaded object); no transformation
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
- v2 register trusts only the extension and the object's size — bytes never reach the backend, so `validate_document_content` sniffing (v1 only) is skipped. Register verifies the object exists at `{storage_path}/{document_id}` and deletes it when oversized.
- Collections are immutable-ish: deletion semantics in deep dive §10.
- OpenAI file-batch id: the SDK's `file_batches.poll()` / `upload_and_poll()` final return deserializes a vector-store body, so its `.id` is the `vs_` id, not the `vsfb_` batch id. `crud/rag/open_ai.py` captures the batch id from `create()` before polling and uses it for `list_files`. Any failed file is a hard failure (whole vector store rolled back); partial indexing needs an add-documents endpoint first.
- The SDK's `file_batches.poll()` never times out. `_poll_file_batch` polls `retrieve` in a loop with no internal deadline — the Celery soft time limit bounds it, and its `SoftTimeLimitExceeded` aborts the task. An earlier version took a deadline from the caller via a `task_budget` `ContextVar` (and a fixed `BATCH_POLL_TIMEOUT_SECONDS`); both were deleted. Don't reintroduce caller coupling here.
- Retries are a **single tenacity layer** wrapping `_create_and_index_batch` (create + poll + validate) in `crud/rag/open_ai.py`: `stop_after_attempt(BATCH_INDEX_MAX_ATTEMPTS)` (3 retries) with exponential backoff (~2s/4s/8s), retrying on `OpenAIError` or `RuntimeError` (indexing error, failed files, or non-`completed` status). SDK-level retries are **off** (`max_retries=0` in `providers/registry.py`) so nothing stacks. `SoftTimeLimitExceeded` is deliberately *not* retried (neither `OpenAIError` nor `RuntimeError`), so a spent window aborts immediately. History: a prior tenacity attempt that *stacked on top of* SDK retries measured 4.5x slower (3×3=9 requests/call) — the fix was to make tenacity the sole layer, not to drop it. `upload_files` runs once per task (outside the retry); only the create+attach+index is retried, so retries re-attach already-uploaded file IDs rather than re-uploading.
- `OPENAI_TIMEOUT_SECONDS` (30s) is a **stall detector, not an upload deadline**. httpx has no total-request timeout — connect/read/write/pool are each per-socket-operation — so a big document that streams steadily uploads fine however long it takes; the timeout only fires after 30 consecutive seconds of zero bytes. With SDK retries off, one hung call is capped near 30s, and tenacity's backoff keeps the whole batch inside the soft-limit window. Verified: a 10MB body taking 12s under a 2s write timeout succeeds, and only a stalled receiver raises `WriteTimeout`. Don't raise it out of fear that large files get cut off — they don't.
- A batch failure (indexing error, failed files, cancelled/failed status, timeout) is retried **in-task** by the tenacity layer above — 3 retries, exponential backoff, all inside one Celery soft-time-limit window. There is **no** Celery-level re-queue: if the batch (or its retries) can't finish within the window, `SoftTimeLimitExceeded` fires and the job is marked FAILED (`_handle_job_failure`), not re-queued. Trade-off: a collection whose batch genuinely needs more than one window fails rather than resuming across windows — size batches to fit. Setup does not retry either (a second `create_vector_store()` orphans the first).
- `documents_uploaded` is deduped on append — a task re-run (e.g. redelivered after worker loss, since `acks_late` is on) re-adds its own IDs, and `DocumentCrud.read_each` raises when duplicates collapse in its `IN` clause.
- Do not "fix" batch timing with a self-requeueing continuation task — a task that returns normally is not redelivered, and a lost continuation strands the job in `PROCESSING`, which nothing monitors or recovers. Deep dive §11.5.
- Uploads happen per batch in `execute_batch_job`, not in setup; setup only plans batches and creates the vector store.
