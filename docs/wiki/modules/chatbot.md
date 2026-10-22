# Module: Chatbot

Prompt-driven chatbot ("Beneficiary Info Extractor") over a two-node LangGraph and a LangChain chat model (Anthropic). The admin's free-text `system_prompt` defines the goal, the fields to collect, and the rules; each turn extracts the fields the user has given so far and then replies. The model and reasoning settings come from a saved Kaapi config (`config_id` + optional `config_version`, latest if omitted); only Anthropic text completion configs are supported. Two entry points: a stateless synchronous turn (caller resends `system_prompt` + full `history`) and server-side sessions whose turns run on Celery and reply via webhook. No deep-dive doc yet.

All paths relative to `backend/app/`.

## Routes
- `api/routes/chatbot/message.py` — `POST /chatbot/message` (one synchronous stateless turn → `ChatbotTurnResponse{messages, collected_fields, completion_status}`)
- `api/routes/chatbot/session.py`
  - `POST /chatbot/sessions` (create; validates the config → `{session_id}`)
  - `GET /chatbot/sessions?contact_ref=` (latest session for a contact in the caller's project → `ChatbotSessionSummary`, incl. `status`, `completion_status`, `collected_fields`)
  - `POST /chatbot/sessions/{session_id}/message` (stores the user message, enqueues the turn → 202; reply POSTed to `webhook_url`)
- All require project scope.

## Tables (SQLModel, `models/chatbot.py`)
- `chatbot_session` (`ChatbotSession`) — `name`, `contact_ref`, `config_id` (FK `config`, RESTRICT) / `config_version`, `system_prompt`, `status` (`ChatbotSessionStatusEnum` active/inactive = session in use), `completion_status` (`ChatbotCompletionStatusEnum` in_progress/complete = all required fields collected), `collected_fields` (JSON flat `{field_name: value}`), org/project FKs. Migration `088`.
- `chatbot_session_message` (`ChatbotSessionMessage`) — ordered transcript rows (`role` user/assistant; the prompt lives on the session).
- Request/response shapes: `ChatbotMessage`, `ChatbotTurnRequest`, `ChatbotTurnResponse`, `CreateChatbotSessionRequest/Response`, `ChatbotSessionSummary`, `SendChatbotSessionMessageRequest/Response`.

## Turn flow (onboarding agent, `services/chatbot/agents/onboarding/graph.py`)
1. `extract_fields_node` — skipped until there is a real user turn (opening greeting). Otherwise calls the model with structured output (`FieldExtraction{collected_fields, missing_fields, is_complete}`) over the admin prompt, the previously collected map, and the (summarized) conversation. Uses Anthropic native structured output (`with_structured_output(..., method="json_schema")`) with the config's own thinking/effort, because forced tool use is rejected with thinking on and Opus 5.5 can't disable thinking. Merged in code: `{**prior, **extracted}`; `is_complete` only when nothing is missing.
2. `chatbot_node` — replies using the same summarized view plus a system note listing collected/missing fields, so it asks only for what's missing and wraps up when complete.
3. `run_chatbot_turn(..., collected_fields=prior)` → `ChatbotTurnResult{transcript, collected_fields, is_complete}` (`.completion_status` maps to the enum). Anthropic errors from either call, and langchain_anthropic's client-side model-config `ValueError`s ("[KAAPI] Unsupported model configuration"), map to `ChatbotProviderError`, so the session path always sends a failure webhook.

## Services / CRUD
- `services/chatbot/agents/onboarding/graph.py` — `run_chatbot_turn` (public entry point), `build_onboarding_graph`, the extraction/reply nodes
- `services/chatbot/agents/onboarding/prompts.py` — extraction instruction/templates, progress-note templates, `EXTRACTION_METHOD`
- `services/chatbot/agents/onboarding/types.py` — `OnboardingState`, `FieldExtraction`
- `services/chatbot/history.py` — agent-agnostic history plumbing: `summarize_if_needed`, `to_langchain_messages` / `to_message_dicts`, kickoff message, `has_user_turn`, `insert_after_leading_system`
- `services/chatbot/types.py` — shared `ChatbotTurnResult`, `ChatbotMessageDict`
- `services/chatbot/session.py` — `process_chatbot_session_message` (Celery worker body: load session + history, run the turn with the stored `collected_fields`, store the reply and update `collected_fields` / `completion_status`, POST the webhook)
- `services/chatbot/config.py` — `resolve_chatbot_model_settings` (saved config + request overrides → `ChatbotModelSettings`; 404 via `ConfigVersionCrud`, 422 for non-Anthropic/non-text/no-model configs)
- `services/chatbot/utils.py` — `ChatbotModelSettings`, defaults (`DEFAULT_MAX_OUTPUT_TOKENS`, `DEFAULT_HISTORY_TOKEN_LIMIT`, `FLOW_KEEP_RECENT_MESSAGES`), `get_chat_model`
- `crud/chatbot.py` — `create_chatbot_session`, `get_chatbot_session`, `get_latest_chatbot_session_by_contact_ref`, `update_chatbot_session_progress`, `create_chatbot_session_message`, `list_chatbot_session_messages`
- `crud/config/version.py` — `ConfigVersionCrud.read_latest` (used when `config_version` is omitted)

## Async
- `celery/tasks/job_execution.py` — `run_chatbot_session_message` (enqueued via `celery/utils.py:start_chatbot_session_message`) → `process_chatbot_session_message`.
- Webhook success payload: `{"success": true, "data": {"session_id", "reply", "status", "completion_status", "collected_fields"}}`; failure: `{"success": false, "error", "data": {"session_id"}}`.

## External
- Anthropic via `langchain-anthropic` (`ANTHROPIC_API_KEY` setting) — two calls per turn once the user has spoken (extraction + reply), plus one summary call when history exceeds the token budget.
- Caller webhook (`send_callback`, HMAC-signed with the project webhook secret).
