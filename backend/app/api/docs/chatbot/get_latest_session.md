Get the most recently created chatbot session for the given `contact_ref` in the caller's project.

`contact_ref` is matched exactly against the value supplied when the session was created. The response carries the session's current `status` (`active`/`inactive`), its `completion_status` (`in_progress` until every field the system prompt asks for is collected, then `complete`), and the flat `collected_fields` map (`{field_name: value}`) gathered so far. Returns 404 when the contact has no sessions in this project.
