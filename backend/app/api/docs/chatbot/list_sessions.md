List every chatbot session for the given `contact_ref` in the caller's project, most recent first.

`contact_ref` is matched exactly against the value supplied when the session was created. Each entry carries the session's current `status` and the `collected_fields` gathered so far. Returns an empty list when the contact has no sessions in this project.
