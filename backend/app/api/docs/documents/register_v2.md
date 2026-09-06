Register a document at the `document_id` issued by `POST /api/v2/documents/uploads`, from the bytes staged at its pre-signed URL.

Final step of the v2 upload flow. Send the filename the upload URL was issued for — its extension determines where the bytes were staged. The staged object is moved to its permanent location, the document row is created, and the response carries a fresh signed URL for reading the file back.

Errors: `400` if nothing is staged for that `document_id` and filename (the upload never happened, lapsed, or the filename differs from the one the URL was issued for) or the extension is unsupported; `413` if the uploaded file exceeds 25 MB, in which case the staged object is deleted; `409` if the `document_id` was already registered — open a new upload session in that case.

The 25 MB cap is enforced here rather than at upload time: a pre-signed PUT cannot enforce a content-length range (that would require a pre-signed POST, which would break the raw-PUT contract).

Document transformation is not available on v2. Use `POST /api/v1/documents` if you need a `target_format` conversion.
