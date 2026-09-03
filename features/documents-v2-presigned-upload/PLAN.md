# Documents v2 Presigned Upload — Implementation Plan

Source spec: GitHub issue #1169 (verbal description, see Open Questions). Related: closed issue #1143.

## Summary

Add a v2 documents surface that replaces multipart upload through the backend with a two-step presigned-URL flow: the client requests a presigned PUT URL (JSON), uploads the file directly to S3, then registers the document (JSON) which creates the `document` row. Document transformation is not part of v2 upload; it stays a v1-only concern. v1 endpoints stay as they are, not deprecated (user decision). No schema change, no new tables.

## Blast Radius

Primary entities: Document (new write path only, same row shape).

| Surface | Hop | Impact | Decision |
|---|---|---|---|
| Document (table) | 0 | New rows created via v2 register endpoint; identical columns and key layout (`{storage_path}/{document_id}`) | in scope |
| DocumentCollection / Collection | 1 | None, consumes Document rows which keep the same shape | out of scope |
| DocTransformationJob | 1 | Untouched; v2 register does not schedule transformations (user decision) | out of scope |
| FineTuning / ModelEvaluation | 1 | None, read Document rows unchanged | out of scope |
| Object storage (`core/cloud/storage.py`) | ext | New `get_signed_upload_url` method on `CloudStorage` + `AmazonCloudStorage` (presigned `put_object`) | in scope |
| kaapi-frontend console | ext | Keeps using v1 unchanged; migration is a later frontend task | deferred |
| Glific | ext | v2 endpoint docs describe the new flow; v1 untouched | in scope (docs only) |
| Langfuse | ext | Unaffected, no LLM call path touched | out of scope |
| Provider batch APIs | ext | Unaffected | out of scope |

## Steps

### 1. Core: presigned PUT support in storage
- Files: `backend/app/core/cloud/storage.py` (change)
- Add abstract `get_signed_upload_url(self, key: str, content_type: str | None = None, expires_in: int = 3600) -> str` to `CloudStorage`; implement in `AmazonCloudStorage` via `generate_presigned_url("put_object", ...)`, capping `expires_in` at `MAX_SIGNED_URL_EXPIRY`, logging per convention.
- Depends on: nothing

### 2. Model: v2 request/response schemas
- Files: `backend/app/models/document.py` (change), `backend/app/models/__init__.py` (change)
- Add non-table SQLModel schemas:
  - `DocumentUploadURLRequest` (`filename: str`)
  - `DocumentUploadURLResponse` (`document_id: UUID`, `upload_url: str`, `expires_in: int`)
  - `DocumentRegisterRequest` (`document_id: UUID`, `filename: str`)
- Response for register reuses existing `DocumentUploadResponse` (its `transformation_job` stays `None` in v2).
- Export new names from `models/__init__.py`.
- Depends on: nothing

### 3. Route: v2 documents endpoints
- Files: `backend/app/api/routes/documents_v2.py` (new), `backend/app/api/docs/documents/upload_url_v2.md` (new), `backend/app/api/docs/documents/register_v2.md` (new)
- Router: `APIRouter(prefix="/documents", tags=["Documents v2"])`, mounted under `/api/v2` (step 4). Both endpoints `application/json`, `require_permission(Permission.REQUIRE_PROJECT)`.
- `POST /documents/upload-url`:
  - Validate filename via `get_file_format` (rejects unsupported extensions early).
  - `document_id = uuid4()`; key `{project.storage_path}/{document_id}` via `SimpleStorageName`, matching v1 key layout.
  - Return `DocumentUploadURLResponse` in `APIResponse`.
- `POST /documents` (register):
  - Validate filename extension via `get_file_format`.
  - Verify object exists at the expected key with `storage.get_file_size_kb` (404 → 400 "file not uploaded"); enforce `MAX_DOC_SIZE_MB` (413, delete oversized object).
  - Reject a `document_id` that already exists in `document` (409) to keep register idempotent-safe.
  - Create `Document` row via `DocumentCrud.update`, return `DocumentUploadResponse` with fresh `get_signed_url`. No transformation scheduling in v2.
- Depends on: steps 1, 2

### 4. Wiring: mount v2 router
- Files: `backend/app/api/main.py` (change)
- Import `documents_v2` router, `api_v2_router.include_router(documents_v2.router)`.
- Depends on: step 3

### 5. v1 endpoints unchanged
- v1 upload is NOT deprecated (user decision); no change to `backend/app/api/routes/documents.py` or `upload.md`.

### 6. Wiki update
- Files: `docs/wiki/modules/knowledge-base.md` (change)
- Routes section gains `api/routes/documents_v2.py` (v2 presigned flow) and notes v1 upload deprecated. No `domain-map.md` change (no entity or edge change).
- Depends on: step 3

### 7. Tests
- Files: `backend/app/tests/api/routes/documents/test_route_document_upload_url_v2.py` (new), `backend/app/tests/api/routes/documents/test_route_document_register_v2.py` (new), `backend/app/tests/core/test_storage.py` or nearest existing storage test (change, presign method)
- See Tests section.
- Depends on: steps 1-4

## Migration

None. No table or column changes.

## Tests

Moto (`mock_aws`) for S3, matching `app/tests/api/routes/documents/` fixtures:

- upload-url: happy path returns `document_id` + `upload_url` + `expires_in`; unsupported extension → 400; missing project permission → 403.
- register: happy path (object pre-put into moto bucket) creates Document row, returns signed URL; object absent → 400; oversized object → 413 and object deleted; duplicate `document_id` → 409; unsupported extension → 400.
- storage: `get_signed_upload_url` returns URL containing the key, expiry capped at `MAX_SIGNED_URL_EXPIRY`.

## Open Questions

Assumptions made (issue is short; all inferred, flagged here):

- Two-endpoint flow (upload-url then register) chosen over one endpoint returning a presigned URL plus a pending DB row, to avoid adding an upload-status column and a migration. Register verifies the object server-side instead.
- Content-type sniffing (`validate_document_content`) is skipped in v2; only extension validation and size enforcement run, since bytes never pass through the backend. Downstream transformers already fail cleanly on malformed content. If sniffing is required, register would stream the first bytes from S3.
- No v1 endpoint is deprecated (user decision); v2 exists alongside v1.
- v2 upload excludes document transformation entirely (user decision); clients needing transforms keep using v1 until a v2 transform story exists.
- `expires_in` for the presigned PUT fixed at 3600s (matches existing signed GET default).
