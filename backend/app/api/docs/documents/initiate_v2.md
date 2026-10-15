Open a v2 upload session: get a URL and form fields to send a document straight to Kaapi's object storage.

Step 1 of the three-step flow:

1. `POST /api/v2/documents/uploads` with the filename — returns a `document_id`, an `upload_url`, and `upload_fields`.
2. Upload the file with a single `multipart/form-data` POST to `upload_url`: send every entry in `upload_fields` as a form field, then the file **last** in a field named `file`. No auth header on this call.
3. `PUT /api/v2/documents/{document_id}` to create the document — no body needed.

The filename is validated here (an unsupported type fails before any upload) and travels with the file, so registration in step 3 uses it automatically. Nothing is persisted at this step — hence the `200` rather than a `201` — the `document_id` only becomes a document once you register it.

Maximum file size is 25 MB, enforced by storage as the file uploads: a larger file is rejected outright with `400 EntityTooLarge` and nothing is stored. `upload_url` is valid for `expires_in` seconds (the effective value after server-side capping, which may be shorter than requested); open a new upload session if it lapses.
