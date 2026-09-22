Register the document at the `document_id` issued by `POST /api/v2/documents/uploads`, from the file uploaded to its pre-signed URL.

Final step of the v2 upload flow, and it takes no request body. The uploaded object is moved to its permanent location, the document row is created with the filename captured at step 1, and the response carries a fresh signed URL for reading the file back.

Errors: `400` if nothing was uploaded for that `document_id` (the upload never happened or the URL lapsed); `409` if the `document_id` was already registered — open a new upload session in that case.

The 25 MB cap is enforced by storage while the file uploads, so an oversized file never reaches this step. Document transformation is not available on v2. Use `POST /api/v1/documents` if you need a `target_format` conversion.
