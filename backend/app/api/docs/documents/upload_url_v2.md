Request a pre-signed URL to upload a document straight to Kaapi's object storage.

Step 1 of the v2 upload flow:

1. `POST /api/v2/documents/upload-url` with the filename — returns a `document_id` and an `upload_url`.
2. `PUT` the raw file bytes to `upload_url` (no auth header, no form encoding).
3. `POST /api/v2/documents` with the same `document_id` and `filename` to register the document.

The filename extension is validated here, so an unsupported file type fails before any upload. Nothing is persisted at this step: the `document_id` only becomes a document once you register it. `upload_url` is valid for `expires_in` seconds; request a new one if it lapses. Maximum file size is 25 MB, enforced at registration.
