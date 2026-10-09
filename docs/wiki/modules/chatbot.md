# Module: Chatbot

v0 prompt-driven chatbot: one LangGraph node over a LangChain chat model (Anthropic). Stateless — the caller resends `system_prompt` + full `history` every turn. The model and reasoning settings come from a saved Kaapi config (`config_id` + optional `config_version`, latest if omitted); only Anthropic text completion configs are supported. No deep-dive doc yet.

All paths relative to `backend/app/`.

## Routes
- `api/routes/chatbot.py` — `POST /chatbot/message` (one synchronous turn; requires project scope)

## Tables (SQLModel)
None of its own. Request/response shapes only: `models/chatbot.py` (`ChatbotMessage`, `ChatbotMessageRoleEnum`, `ChatbotTurnRequest`, `ChatbotTurnResponse`). Reads `config` / `config_version` (see the config module) to resolve the model.

## Services / CRUD
- `services/chatbot/config.py` — `resolve_chatbot_model_settings` (saved config + request overrides → `ChatbotModelSettings`; 404 via `ConfigVersionCrud`, 422 for non-Anthropic/non-text/no-model configs)
- `services/chatbot/engine.py` — `run_chatbot_turn` (dict <-> LangChain message conversion, error mapping), `build_chatbot_graph`, history summarization
- `services/chatbot/utils.py` — `ChatbotModelSettings`, defaults (`DEFAULT_MAX_OUTPUT_TOKENS`, `DEFAULT_HISTORY_TOKEN_LIMIT`, `FLOW_KEEP_RECENT_MESSAGES`), `get_chat_model`
- `crud/config/version.py` — `ConfigVersionCrud.read_latest` (used when `config_version` is omitted)

## Async
- None; synchronous request/response, not Celery.

## External
- Anthropic via `langchain-anthropic` (`ANTHROPIC_API_KEY` setting).
