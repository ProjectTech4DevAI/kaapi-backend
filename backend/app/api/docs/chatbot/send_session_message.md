Queue one turn of a chatbot session; the bot reply is delivered asynchronously to `webhook_url`.

The user `message` is stored immediately and the response is `202` with `status: "queued"`. Once the turn completes, Kaapi POSTs `{"success": true, "data": {"session_id", "reply", "status", "collected_fields"}}` to `webhook_url` (HMAC-signed when the project has a webhook secret). On a model or config failure it POSTs `{"success": false, "error": "...", "data": {"session_id"}}` instead.

On the first call of a session, omit `message` to get the bot's opening greeting; after that `message` is required. `webhook_url` must be a public HTTPS URL. Returns 404 if the session isn't in the caller's project and 422 for an invalid `webhook_url` or a missing `message` on a started session.
