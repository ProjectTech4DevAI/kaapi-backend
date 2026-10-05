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
| Tool registry | `services/agent/tools.py`: `AgentTool`, `READ_ONLY_TOOLS`, `TOOLS_BY_NAME`, `ANTHROPIC_TOOL_DEFINITIONS`, per-tool allowlist specs |
| Projection | `services/agent/projection.py`: `make_projector`, `project_record`, leaf kinds `SCALAR` / `SCALAR_LIST` / `NUMERIC_MAP` |
| Tool executor | `services/agent/executor.py`: `execute_tool_call()` → `ToolCallOutcome` |
| Prompt | `services/agent/prompts.py`: `AGENT_SYSTEM_PROMPT` (stable; it is the prompt-cache prefix) |
| Errors | `services/agent/exceptions.py`: `AgentNotConfiguredError` (→ 503), `AgentLLMError` (→ 502) |
| Settings | `core/config.py`: `AGENT_MODEL`, `AGENT_EFFORT`, `AGENT_MAX_TOKENS`, `AGENT_MAX_ITERATIONS`, `AGENT_TOOL_TIMEOUT_SECONDS`, `AGENT_TOOL_RESULT_MAX_CHARS`, `AGENT_LLM_TIMEOUT_SECONDS`; reuses `ANTHROPIC_API_KEY` |

## Tool registry (extension point)

To add a tool, add one `AgentTool` entry to `READ_ONLY_TOOLS`: `name`, a model-facing
`description` (what it returns, when to use it, how it chains), `method="GET"`, a `path` template
under `settings.API_V1_STR`, a Pydantic `args_model` (`extra="forbid"`, typed path params), a
`projector` built from an allowlist spec (required for every registry tool; checked at import),
optional `fixed_query_params`, and `page_size_param` for list tools (see Pagination). Placeholders in the path template become path params; every other
non-None arg becomes a query param.

v1 tools cover: text eval runs + datasets (`/evaluations`, `/evaluations/datasets`), config versions
(`/configs/{config_id}/versions/{version_number}`), collections, documents, and STT/TTS runs + datasets
(`/evaluations/{stt,tts}/...`).

## Pagination

Routes report "more data" inconsistently (`has_more`, `total/limit/offset`, or nothing), so the harness
computes it uniformly, the way `/configs` does.

- `AgentTool.page_size_param` names the args-model field holding the page size (`"limit"`); `None` means
  not paginated. Checked at definition time to be an args-model field.
- The executor sends `limit + 1` to the route. If the returned `data` is a list longer than `limit`, it is
  trimmed to `limit` and `has_more=true`, else `has_more=false`. Trimming happens before projection and
  truncation, so the probe row never reaches the model. The `tool_calls` log keeps the model's own `limit`.
- Model-facing `limit` is capped at 50 so `limit + 1` stays within the routes' `le=100`.
- The offset arg keeps each route's own name:

| Tool | Route | Offset param | Default limit |
|---|---|---|---|
| `list_evaluation_runs` | `/evaluations` | `offset` | 10 |
| `list_evaluation_datasets` | `/evaluations/datasets` | `offset` | 20 |
| `list_documents` | `/documents` | `skip` | 20 |
| `list_stt_evaluation_runs` / `list_tts_evaluation_runs` | `/evaluations/{stt,tts}/runs` | `offset` | 10 |
| `list_stt_evaluation_datasets` / `list_tts_evaluation_datasets` | `/evaluations/{stt,tts}/datasets` | `offset` | 20 |

`list_collections` is not paginated (the route returns everything). The system prompt tells the model to
keep paging while `has_more` is true and to report a lower bound ("at least N") if it stops early.

## Data minimization

The model only sees fields a tool explicitly allowlists; everything else in a route's response is dropped.

- **Allowlist per tool.** Each tool's `projector` is `make_projector(spec)`, where a spec maps field name →
  leaf kind (`SCALAR`, `SCALAR_LIST`, `NUMERIC_MAP`) or a nested spec (applied to a dict, or to each item
  of a list). Unnamed keys are dropped; a named field whose type doesn't match its kind (e.g. a scalar
  that became an object) becomes `null` instead of leaking nested data. Run tools expose run-level
  aggregates only (summary scores, overall breakdown, category metrics, cost), never per-item traces,
  results or samples, and `get_config_version` exposes only completion type, provider, model and
  `knowledge_base_ids`, never prompt text. No spec names a storage URL.
- **Fixed query params.** `AgentTool.fixed_query_params` are sent on every call and cannot be set by the
  model (a key that overlaps an args-model field fails at definition time). They pin off routes'
  default-on bulk payloads: `include_results=false` on the STT/TTS get-run tools and
  `include_samples=false` on `get_stt_evaluation_dataset`.
- **Envelope.** Route `metadata` is always dropped. Paginated tools get exactly
  `metadata: {"has_more": bool}` (see Pagination); non-paginated tools return bare `data`. A
  `success=false` error string is still passed through.
- **Extension rule.** A new tool is a registry entry plus an allowlist spec covering only the fields the
  model needs to answer questions or chain to the next call.

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
  followed. No tool exposes signed-URL, trace-fetch or resync flags, and responses pass through the
  allowlist projection above.

## Consumes (blast radius)

Response-shape changes to any route in the registry ripple into the agent's allowlist specs and tool
descriptions; a renamed field silently disappears from the agent's view until its spec is updated. Check `services/agent/tools.py` when changing those routes.
