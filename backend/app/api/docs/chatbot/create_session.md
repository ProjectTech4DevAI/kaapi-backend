Start a server-side chatbot conversation and return its `session_id` and `chatbot_ref`.

The `system_prompt` and the `config_id` / `config_version` pin are stored on the session, so later turns only send the new user message. `config_id` must be a config in the caller's project whose completion is `provider: "anthropic"`, `type: "text"` with `params.model` set; when `config_version` is omitted, each turn uses the config's latest version at that time.

`name` is the chatbot's display name only; omit it to get `"Untitled Chatbot"`. `contact_ref` is the caller's own identifier for the end user and is stored as-is.

`chatbot_ref` identifies the chatbot. Omit it to create a new chatbot: Kaapi generates a ref of the form `LLMResponseWithPrompt-<5 hex chars>` (e.g. `LLMResponseWithPrompt-913f0`), unique within the project, and returns it. To start a new session for the same chatbot, send that returned `chatbot_ref` back; it is used as-is.

A contact has at most one active session per chatbot. Creating a session marks the contact's previous `active` sessions with the same `chatbot_ref` as `inactive`, in the same transaction; those still `in_progress` become `abandoned`, while `complete` ones stay `complete`. Sessions of other chatbots or other contacts are untouched. Returns 404 if the config or version isn't found in the project, 422 if it isn't a usable Anthropic text config, 409 if another session for the same `contact_ref` + `chatbot_ref` was created concurrently (retry the request), and 503 in the unlikely case no unused `chatbot_ref` could be generated (retry).
