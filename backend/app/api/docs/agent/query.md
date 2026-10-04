Ask a natural-language question about your own project's data — evaluation runs and datasets (text, STT, TTS), config versions, collections and documents — and get a plain-language answer.

The agent is **read-only**: it answers by calling Kaapi's own GET endpoints with your credentials, so it only sees what you could fetch yourself and cannot create, change or delete anything. `tool_calls` lists every lookup it made; `stop_reason` is `iteration_limit` when it ran out of lookups before finishing (the answer may be partial) and `max_tokens` when the answer was cut off.

Returns `503` when the agent is not configured on this server and `502` when the underlying model call fails.
