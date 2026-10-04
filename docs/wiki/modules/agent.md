# Module: Agent (read-only natural-language Q&A)

`POST /api/v1/agent`: the caller asks a question about their own project's data and Claude answers
by calling Kaapi's own GET routes in a LangGraph tool-use loop. No tables, no migration, no Celery,
no checkpointer (v1 is single-turn and returns a synchronous response).

All paths relative to `backend/app/`.

| Surface | Path |
|---|---|
| Route | `api/routes/agent.py` (`query_agent`, `_extract_forwardable_credentials`); swagger `api/docs/agent/query.md` |
| Models (non-table) | `models/agent.py`: `AgentQueryRequest`, `AgentQueryResponse`, `AgentToolCallPublic`, `AgentUsagePublic`, `AgentStopReasonEnum` |
| Graph | `services/agent/graph.py`: `agent_node` → `route_after_agent` → `tool_executor_node` → back; `build_agent_graph()` (cached), `run_agent_query()` |
| State / context | `services/agent/state.py`: `AgentState` (JSON-serializable, checkpoint-ready), `AgentContext` (runtime-only: credentials + clients) |
| Tool registry | `services/agent/tools.py`: `AgentTool`, `READ_ONLY_TOOLS`, `TOOLS_BY_NAME`, `ANTHROPIC_TOOL_DEFINITIONS`, projectors |
| Tool executor | `services/agent/executor.py`: `execute_tool_call()` → `ToolCallOutcome` |
| Prompt | `services/agent/prompts.py`: `AGENT_SYSTEM_PROMPT` (stable; it is the prompt-cache prefix) |
| Errors | `services/agent/exceptions.py`: `AgentNotConfiguredError` (→ 503), `AgentLLMError` (→ 502) |
| Settings | `core/config.py`: `AGENT_MODEL`, `AGENT_EFFORT`, `AGENT_MAX_TOKENS`, `AGENT_MAX_ITERATIONS`, `AGENT_TOOL_TIMEOUT_SECONDS`, `AGENT_TOOL_RESULT_MAX_CHARS`, `AGENT_LLM_TIMEOUT_SECONDS`; reuses `ANTHROPIC_API_KEY` |

## Tool registry (extension point)

To add a tool, add one `AgentTool` entry to `READ_ONLY_TOOLS`: `name`, a model-facing
`description` (what it returns, when to use it, how it chains), `method="GET"`, a `path` template
under `settings.API_V1_STR`, a Pydantic `args_model` (`extra="forbid"`, typed path params), and an
optional `projector` that trims the response `data`. Placeholders in the path template become path params;
every other non-None arg becomes a query param.

v1 tools cover: text eval runs + datasets (`/evaluations`, `/evaluations/datasets`), config versions
(`/configs/{config_id}/versions/{version_number}`), collections, documents, and STT/TTS runs + datasets
(`/evaluations/{stt,tts}/...`).

## Security model

- Tools call the real routes in-process (`httpx.ASGITransport` over `request.app`), so every route's
  auth, tenant scoping, validation and serialization still apply. The agent can only see what the
  caller could fetch directly.
- Read-only is enforced by the harness: `ALLOWED_TOOL_METHODS = {"GET"}` is checked when a tool is
  defined and again before sending. Unknown tool names return an error result and no request is sent.
- The caller's credential (X-API-KEY / Bearer header / `access_token` cookie, same priority as
  `get_auth_context`) is carried only in LangGraph runtime context. It is never stored in graph state or
  messages, never logged, and never shown to the model. Only credential headers and `X-Request-ID` are sent.
- Model-supplied values never become headers. Path params are type-validated and percent-encoded, and redirects are not
  followed. No tool exposes signed-URL, trace-fetch or resync flags, and projectors drop storage URLs and per-item score traces.

## Consumes (blast radius)

Response-shape changes to any route in the registry ripple into the agent's projectors and tool
descriptions. Check `services/agent/tools.py` when changing those routes.
