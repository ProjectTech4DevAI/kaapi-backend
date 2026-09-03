Register a document that was uploaded to the pre-signed URL from `POST /api/v2/documents/upload-url`.

Final step of the v2 upload flow. Pass the `document_id` returned by the upload URL endpoint together with the filename; the document row is created and the response carries a fresh signed URL for reading the file back.

Errors: `400` if no file was uploaded for that `document_id` (or the extension is unsupported), `413` if the uploaded file exceeds 25 MB (the object is deleted), `409` if the `document_id` was already registered — request a new upload URL in that case.

Document transformation is not available on v2. Use `POST /api/v1/documents` if you need a `target_format` conversion.
