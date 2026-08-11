# Module: LLM Call

Live LLM invocation: `POST /llm/call` and chains, driven by `LLMCallConfig` (saved or ad-hoc), with guardrails and provider routing.
Deep dive: `docs/architecture/kaapi-llm-call-ARCHITECTURE.md` (§3 request/config anatomy, §5 pipeline, §6 provider routing, §7 guardrails, §8 persistence).

All paths relative to `backend/app/`.

## Routes
- `api/routes/llm.py` — `POST /llm/call`
- `api/routes/llm_chain.py` — chains
- `api/routes/llm_sts.py` — speech-to-speech
- `api/routes/config/config.py`, `api/routes/config/version.py` — saved config CRUD + versions
- `api/routes/guardrails.py` — guardrail validators

## Tables (SQLModel)
| Table | Model |
|---|---|
| `llm_call` (LlmCall), `llm_chain` (LlmChain) | `models/llm/request.py` — `metadata` JSONB (attr `call_metadata`): `{effective_provider}` stamped after routing, NULL pre-076 |
| `config` (Config) | `models/config/config.py` |
| `config_version` (ConfigVersion) | `models/config/version.py` |

## Key pydantic/SQLModel schemas (`models/llm/request.py`)
- `LLMCallConfig` — one-of: saved reference (`id` + `version`) XOR ad-hoc `blob` (validator-enforced)
- `ConfigBlob` — `completion` + optional `prompt_template` (`PromptTemplate.template`, plain string; `{{input}}` interpolation is llm-chain-only) + `input_guardrails`/`output_guardrails`
- `CompletionConfig` — discriminated union on `provider`: `KaapiCompletionConfig` (standardized params: `TextLLMParams`/`STTLLMParams`/`TTSLLMParams`), `NativeCompletionConfig` (pass-through), `ProxyCompletionConfig` (client's own endpoint)
- `QueryParams` — per-call input + `ConversationConfig`
- `models/guardrails/` — validator config shapes

## Services / CRUD
- `services/llm/` — `mappers.py` (Kaapi params → provider API), `providers/`, `chain/`, `guardrails.py`, `jobs.py`
- `services/llm/providers/` — `registry.py` routes `google`/`google-native` by `GEMINI_DEFAULT_INFERENCE_ROUTE` (`aistudio` | `gcp`); explicit `google-aistudio` → `google_aistudio.py` (`GoogleAIProvider`), `google-gcp` → `google_gcp.py` (`GoogleGCPProvider`, Vertex AI REST). Flows: `backend/docs/gemini-routing-flows.md`
- `services/guardrails/` — validator execution
- `crud/llm.py`, `crud/llm_chain.py`, `crud/config/` — persistence

## Async
- Job execution via Celery: `celery/tasks/job_execution.py`; `job` table tracks state.

## External
- OpenAI / Gemini / Anthropic (via `services/llm/providers/`), proxy endpoints, Langfuse tracing.

## Gotchas
- Saved config resolution = `config_id` + `version` pinned; ad-hoc blob never persisted.
- `type=proxy` auto-injects `provider="proxy"` (ConfigBlob validator).
- Feature needs an LLM config? Spec `LLMCallConfig` whole (never a bespoke params + prompt pair), and prefer an optional per-request field over a per-project binding table — saved references already give durable versioned config via `config`/`config_version`.
- `PromptTemplate.template` is a plain prompt string; `{{input}}` interpolation is llm-chain-only — features that assemble their own inputs don't use it.
