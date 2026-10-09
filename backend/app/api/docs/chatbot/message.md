Run one turn of a prompt-driven chatbot conversation and return the full updated transcript.

The chat model comes from a saved Kaapi config: `config_id` must be a config in the caller's project whose completion is `provider: "anthropic"`, `type: "text"` with `params.model` set. `config_version` picks a version (latest if omitted). The config's `params.effort` sets reasoning effort (provider default if absent) and `params.thinking` overrides the default adaptive thinking. `max_tokens` (default 1024) and `history_token_limit` (default 4000; older turns are summarized above it) are optional per-request overrides.

There is no server-side session yet: send the admin-authored `system_prompt` on every call, and send the previous response's `messages` back as `history`. The returned `messages` contain the system prompt followed by every user/assistant turn so far, ending with the new assistant reply. Any `system` entries in `history` are ignored in favour of `system_prompt`.

On the very first call, send an empty `history` and omit `message` to get the bot's opening greeting. Once `history` is non-empty, `message` is required. Returns 404 if the config or version isn't found in the project, 422 for a missing `message` or a config that isn't a usable Anthropic text config, and 502 for provider failures.
