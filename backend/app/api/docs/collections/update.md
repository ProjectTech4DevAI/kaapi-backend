Update the editable fields of an existing collection.

You can update the collection's `name` and/or `description`. Both fields are optional in the request body — only the fields you include will be updated. Fields you omit (or send as `null`) are left unchanged.

**Behavior:**

- `name`: Must be unique within the project. If the new name is already in use by another active collection in the same project, the request fails with `409 Conflict`. Sending the same name the collection already has is a no-op (no conflict raised).
- `description`: Free-form text. Send an empty string `""` to clear it.

Other collection fields (such as `llm_service_id`, `provider`, documents, etc.) cannot be modified through this endpoint.
