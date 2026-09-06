Open a v2 upload session: get a pre-signed URL to send a document straight to Kaapi's object storage.

Step 1 of the three-step flow:

1. `POST /api/v2/documents/uploads` with the filename — returns a `document_id` and an `upload_signed_url`.
2. `PUT` the raw file bytes to `upload_signed_url` — the body is the file itself, with no auth header, no form encoding, and no extra headers.
3. `PUT /api/v2/documents/{document_id}` with the same filename to create the document.

The filename extension is validated here, so an unsupported file type fails before any upload. Nothing is persisted at this step — hence the `200` rather than a `201` — the `document_id` only becomes a document once you register it in step 3. The filename sent in step 3 must be the same one this URL was issued for: its extension determines where the bytes are staged, so a different extension will find nothing to register.

`upload_signed_url` is valid for `expires_in` seconds (the effective value after server-side capping, which may be shorter than requested); open a new upload session if it lapses. Maximum file size is 25 MB, enforced at registration rather than at upload time — a pre-signed PUT cannot enforce a content-length range (that would require a pre-signed POST, which would break the raw-PUT contract).
