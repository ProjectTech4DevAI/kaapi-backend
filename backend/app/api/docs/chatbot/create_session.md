Start a server-side chatbot conversation and return its `session_id`.

The `system_prompt` and the `config_id` / `config_version` pin are stored on the session, so later turns only send the new user message. `config_id` must be a config in the caller's project whose completion is `provider: "anthropic"`, `type: "text"` with `params.model` set; when `config_version` is omitted, each turn uses the config's latest version at that time.

`contact_ref` is the caller's own identifier for the end user and is stored as-is. Returns 404 if the config or version isn't found in the project and 422 if it isn't a usable Anthropic text config.
