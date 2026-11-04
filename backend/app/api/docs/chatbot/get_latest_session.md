Get the most recently created chatbot session for `contact_ref` in the caller's project, optionally narrowed to one chatbot with `chatbot_id`.

`contact_ref` is required and matched exactly. Omit `chatbot_id` to get the contact's newest session across the project's chatbots — the response's `chatbot_id` and `chatbot_name` say which one. Sessions of deleted chatbots are not returned.

The response carries the pinned `flow_version`, the `active_node_id` and that node's `prompt`, the session `status` (`active`, or `inactive` once a newer session for the same contact and chatbot was created or the chatbot was deleted), its `completion_status` (`in_progress` until every field the prompt asks for is collected, then `complete`; `abandoned` if it was deactivated while still `in_progress`), and the flat `collected_fields` map (`{field_name: value}`) gathered so far. Returns 404 when no matching session exists in this project.
