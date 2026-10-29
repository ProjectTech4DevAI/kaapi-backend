# Module: Chatbot

A project has many chatbots; each chatbot's behaviour is a React Flow graph (nodes + edges JSON) edited in the UI and versioned with draft/publish, so editing never breaks running conversations. Agent nodes carry their own `prompt` and saved Kaapi config (`config_id` + `config_version`, pinned at publish); only Anthropic text completion configs are supported. Conversations are server-side sessions pinned to the live flow version and an active node; turns run on Celery and reply via webhook. On this branch the runtime only executes `onboarding` agent nodes ("Beneficiary Info Extractor" over a two-node LangGraph): the node prompt defines the goal, fields to collect and rules; each turn extracts the fields given so far and then replies. No deep-dive doc yet.

All paths relative to `backend/app/`.

## Routes (`/chatbots`, all require project scope and the `CHATBOT` feature flag)
- `api/routes/chatbot/__init__.py` mounts session routes first so literal `/sessions/...` paths are matched before `/{chatbot_id}/...`. `require_feature(FeatureFlag.CHATBOT)` sits on the router → 403 when off. Migration `089` only adds the enum value — no rows are seeded, so the flag is off until a superuser creates it per project via `POST /feature-flags` (`enabled: true`).
- `api/routes/chatbot/chatbot.py`
  - `POST /chatbots` {name} → 201 `ChatbotPublic`
  - `GET /chatbots?skip=&limit=` → the project's non-deleted chatbots, newest `updated_at` first
  - `GET /chatbots/{chatbot_id}`, `PATCH /chatbots/{chatbot_id}` {name} → 404 when unknown/deleted
  - `DELETE /chatbots/{chatbot_id}` → 200 `Message`; soft delete + deactivates the chatbot's active sessions in the same transaction
- `api/routes/chatbot/flow.py` (response `ChatbotFlowVersionPublic` {id, chatbot_id, version, is_live, nodes, edges, inserted_at, updated_at})
  - `GET /chatbots/{chatbot_id}/flow?version=` → that version; default the draft if one exists, else live; 404 if no versions
  - `PUT /chatbots/{chatbot_id}/flow` {nodes, edges} → first save creates v1 already live (strict publish validation + pin, 422 on failure); later saves upsert the draft (shape-only validation); 409 on a concurrent version insert
  - `POST /chatbots/{chatbot_id}/flow/publish` → 409 without a draft, 422 listing every strict-validation problem
- `api/routes/chatbot/session.py`
  - `POST /chatbots/{chatbot_id}/sessions` {contact_ref} → 201 {session_id, chatbot_id, flow_version, active_node_id}; 404 unknown/deleted chatbot; 409 no live flow or concurrent create
  - `GET /chatbots/sessions/latest?contact_ref=&chatbot_id=` → `ChatbotSessionSummary` (incl. `chatbot_name`, `flow_version`, `active_node_id`, the active node's `prompt`); `chatbot_id` omitted = newest across the project's non-deleted chatbots
  - `POST /chatbots/sessions/{session_id}/message` → stores the user message with `node_id = active_node_id`, enqueues the turn → 202; 409 if the session is inactive

## Tables (SQLModel, `models/chatbot.py`, migration `088`)
- `chatbot` (`Chatbot`) — `name`, org/project FKs (CASCADE), `deleted_at` soft delete (sessions reference it with RESTRICT).
- `chatbot_flow_version` (`ChatbotFlowVersion`) — `chatbot_id` (CASCADE), `version` (UNIQUE with chatbot_id, `uq_chatbot_flow_version_chatbot_version`), `nodes` / `edges` JSONB stored as the UI sent them (React Flow camelCase keys, extra UI keys round-trip), `is_live` with partial unique index `uq_chatbot_flow_version_live_chatbot` (`WHERE is_live`). The highest version is the draft when not live; live versions are immutable.
- `chatbot_session` (`ChatbotSession`) — `chatbot_id` (RESTRICT), `flow_version_id` (RESTRICT, pinned at create), `contact_ref`, `active_node_id` (React Flow node id inside the pinned flow), `status` (`ChatbotSessionStatusEnum` active/inactive), `completion_status` (`ChatbotCompletionStatusEnum` in_progress/complete/abandoned), `collected_fields` (JSON flat `{field_name: value}`). No org/project columns: tenant scoping joins through `chatbot`. Indexes: `(chatbot_id, contact_ref)`, `contact_ref` (cross-chatbot latest lookup), partial unique `uq_chatbot_session_active_chatbot_contact` on `(chatbot_id, contact_ref) WHERE status = 'active'`.
- `chatbot_session_message` (`ChatbotSessionMessage`) — ordered transcript rows (`role` user/assistant), `node_id` = the flow node that handled the turn.
- Session lifecycle: one active session per contact + chatbot. `active`+`in_progress` → `active`+`complete` as fields are collected. Creating a new session for the same chatbot + `contact_ref` (same transaction as the insert) — or deleting the chatbot — moves active ones to `inactive`+`complete`, or `inactive`+`abandoned` if still in progress. Inactive sessions reject new messages (409); a turn already queued is skipped by the worker, which posts a failure webhook instead. A concurrent double-create hits the partial unique index; crud raises `ChatbotSessionConflictError` → 409.

## Flow JSON (Pydantic in `models/chatbot.py`)
- `ChatbotFlowNode` {id (1..255), type `ChatbotFlowNodeTypeEnum` (input/output/coordinator/onboarding/rag), position {x, y}, data}; agent nodes (`AGENT_NODE_TYPES`) validate `data` as `ChatbotAgentNodeData` {config_id: UUID?, config_version: int ≥ 1?, prompt?}. `ChatbotFlowEdge` {id, source, target, sourceHandle?, targetHandle?, data} — `data.intent` / `data.description` are reserved for coordinator routing. Extra keys allowed everywhere; dumped by alias.
- The first save goes live directly (validated + pinned like a publish); later draft saves validate shape only. Publish (`services/chatbot/flow.py:validate_flow_for_publish`) is strict: unique node ids; ≤ 1 input and ≤ 1 output node; edge endpoints exist; coordinator/rag rejected as not supported yet; every onboarding node has `config_id` + non-empty `prompt` and its config passes `get_chatbot_config_version` + `build_chatbot_model_settings`; a null `config_version` is resolved to the latest and written back into the stored node; an entry node must resolve.
- Entry node (`resolve_entry_node`): no edges → the single agent node; otherwise the agent node targeted by the input node's only outgoing edge.

## Turn flow (onboarding agent, `services/chatbot/agents/onboarding/graph.py`)
1. `extract_fields_node` — skipped until there is a real user turn (opening greeting). Otherwise calls the model with structured output (`FieldExtraction{collected_fields, missing_fields, is_complete}`) over the node prompt, the previously collected map, and the (summarized) conversation. Uses Anthropic native structured output (`with_structured_output(..., method="json_schema")`) with the config's own thinking/effort, because forced tool use is rejected with thinking on and Opus 5.5 can't disable thinking. Merged in code: `{**prior, **extracted}`; `is_complete` only when nothing is missing.
2. `chatbot_node` — replies using the same summarized view plus a system note listing collected/missing fields, so it asks only for what's missing and wraps up when complete.
3. `run_chatbot_turn(..., collected_fields=prior)` → `ChatbotTurnResult{transcript, collected_fields, is_complete}` (`.completion_status` maps to the enum). Anthropic errors from either call, and langchain_anthropic's client-side model-config `ValueError`s ("[KAAPI] Unsupported model configuration"), map to `ChatbotProviderError`, so the session path always sends a failure webhook.

## Services / CRUD
- `services/chatbot/flow.py` — `get_chatbot_flow`, `save_chatbot_flow_draft`, `publish_chatbot_flow` (row-lock the chatbot via `get_chatbot(for_update=True)` to serialize draft saves and publishes), `validate_flow_for_publish`, `resolve_entry_node`, `find_flow_node`, `parse_flow_nodes/edges`, `dump_flow_nodes/edges`, `build_chatbot_flow_version_public`
- `services/chatbot/session.py` — `start_chatbot_session` (live version + entry node → `create_chatbot_session`), `get_latest_chatbot_session_summary`, `process_chatbot_session_message` (Celery worker body: load session + pinned flow version + active node; inactive session, missing node or a non-onboarding node → failure webhook; otherwise run the turn with the node prompt/config and stored `collected_fields`, store the reply with `node_id`, update progress, POST the webhook)
- `services/chatbot/agents/onboarding/{graph,prompts,types}.py` — `run_chatbot_turn`, the extraction/reply nodes, `FieldExtraction`
- `services/chatbot/history.py` — agent-agnostic history plumbing: `summarize_if_needed`, `to_langchain_messages` / `to_message_dicts`, kickoff message, `has_user_turn`, `insert_after_leading_system`
- `services/chatbot/types.py` — shared `ChatbotTurnResult`, `ChatbotMessageDict`
- `services/chatbot/config.py` — `get_chatbot_config_version` (requested or latest version), `build_chatbot_model_settings` (version row → `ChatbotModelSettings`; 422-class `ChatbotInputError` for non-Anthropic/non-text/no-model configs or a bad thinking budget), `resolve_chatbot_model_settings` (both, used by the worker)
- `services/chatbot/utils.py` — `ChatbotModelSettings`, defaults, `get_chat_model`
- `services/chatbot/exceptions.py` — `ChatbotInputError`, `ChatbotProviderError`, `ChatbotSessionConflictError`, `ChatbotNotFoundError`, `ChatbotFlowNotFoundError`, `ChatbotFlowConflictError`, `ChatbotFlowValidationError`
- `crud/chatbot.py` — chatbot: `create_chatbot`, `get_chatbot`, `list_chatbots`, `update_chatbot`, `delete_chatbot`; flow: `get_latest_chatbot_flow_version`, `get_live_chatbot_flow_version`, `get_chatbot_flow_version`, `get_chatbot_flow_version_by_id`, `create_chatbot_flow_version`, `update_chatbot_flow_version_graph`, `publish_chatbot_flow_version` (unsets the previous live before setting the draft live, one commit); session: `create_chatbot_session` (supersedes via `_deactivate_active_chatbot_sessions`), `get_chatbot_session`, `get_latest_chatbot_session` (joined with chatbot + flow version), `update_chatbot_session_progress`, `create_chatbot_session_message`, `list_chatbot_session_messages`

## Async
- `celery/tasks/job_execution.py` — `run_chatbot_session_message` (enqueued via `celery/utils.py:start_chatbot_session_message`) → `process_chatbot_session_message`.
- Webhook success payload: `{"success": true, "data": {"session_id", "chatbot_id", "node_id", "reply", "status", "completion_status", "collected_fields"}}`; failure: `{"success": false, "error", "data": {"session_id"}}`.

## External
- Anthropic via `langchain-anthropic` (`ANTHROPIC_API_KEY` setting) — two calls per turn once the user has spoken (extraction + reply), plus one summary call when history exceeds the token budget.
- Caller webhook (`send_callback`, HMAC-signed with the project webhook secret).
- kaapi-frontend React Flow editor reads/writes the flow JSON as-is.
