# Kaapi Domain Map

**Snapshot: 2026-06-25.** A product-surface map for *blast-radius / impact analysis* — the
source of truth for "if I change surface X, what else is affected?". It maps **product
surfaces** (not individual tables), and the **consumes / consumed-by** edges between them.
Every edge cites its primary-source evidence (a foreign key, a cross-surface import, or an
architecture-doc line) so it can be re-verified fast.

> **Reconcile before you trust it.** A hand-authored map drifts as code changes, and a stale
> map gives *false confidence* — worse than none. When you use this for an SRD's Blast Radius,
> re-derive the primary surface's `Consumed by` edges against live
> `backend/app/{models,crud,services,api/routes}` (grep the FK targets and cross-surface
> imports) and **flag any drift in the SRD**. Treat this file as a checklist to verify against
> code, not an oracle.

## Surfaces

Tiers: **T1** foundational (consumed by nearly everything), **T2** core business logic,
**T3** peripheral, **T4** cross-cutting substrate.

| # | Surface | Backed by | Consumes | Consumed by | Tier |
|---|---|---|---|---|---|
| 1 | **Auth & Tenancy** | `organization`, `project`, `user`, `user_project`, `api_key`, `onboarding` | — | **everything** (every table is `organization_id`/`project_id` scoped) | T1 |
| 2 | **Config Store** *(shared)* | `models/config/`, `model_config` | Auth & Tenancy | LLM Call, Evaluations-Text, Assessment | T1 |
| 3 | **Credentials** | `credentials` (per-project provider creds) | Auth & Tenancy | LLM Call, Knowledge Base, Evaluations (all), Assessment, Response | T1 |
| 4 | **LLM Call** (`/llm/call`) | `services/llm/`, `job`, `llm_call`, `models/llm/` | Config Store, Credentials, Knowledge Base, Guardrails svc, Langfuse | Evaluations-Text (fast-evals), Assessment, Response | T2 |
| 5 | **Knowledge Base** | `document`, `collection`, `document_collection`, `collection_job`, `file`, `doctransform` | Auth & Tenancy, Credentials, S3 | LLM Call (`knowledge_base_ids` → file_search), Evaluations-Text | T2 |
| 6 | **Evaluations — Text** | `evaluation_dataset` (type=text), `evaluation_run`, `services/evaluations/` | Config Store, Knowledge Base, Credentials, Langfuse, Batch infra, LLM Call (fast-evals) | Notifications (completion email) | T2 |
| 7 | **Evaluations — STT** | `stt_sample`, `stt_result`, `services/stt_evaluations/` | Auth & Tenancy, Credentials (Gemini), Batch infra, File store | — | T2 |
| 8 | **Evaluations — TTS** | `tts_result`, `services/tts_evaluations/` | Auth & Tenancy, Credentials (Gemini), Batch infra, S3 | — | T2 |
| 9 | **Assessment** | `assessment`, `assessment_run`, `services/assessment/` | Config Store, **Evaluation Dataset** (`dataset_id` → `evaluation_dataset.id`), Credentials, LLM Call, Batch infra | — | T2 |
| 10 | **Fine-tuning + Model Evaluation** | `fine_tuning`, `model_evaluation` | Auth & Tenancy, Knowledge Base (`document_id` FK) | — | T2 |
| 11 | **Response / Assistants / Conversations** | `assistants`, `openai_conversation`, `response`, `threads`, `message`, `services/response/` | Job tracking, Assistants, OpenAI Conversation, Credentials | — (leaf) | T3 |
| 12 | **Analytics** | `analytics`, `services/analytics/` | LLM Call (`llm_call` audit), Job tracking, Config Store (pricing) | — (read-only) | T3 |
| 13 | **Notifications** | `notification`, `services/notifications/` | Auth & Tenancy, Evaluations-Text, User-Project | — | T3 |
| 14 | **Cross-cutting substrate** | `job`, `batch_job`, `collection_job`, `file`, `feature_flag`, `language`, `core/batch/`, Celery, S3, Langfuse | — | nearly all T2/T3 surfaces | T4 |

## Consumed-by index (the blast-radius lookup)

When a change touches a surface, these are the surfaces that depend on it — your **1-hop**
impact set. Traverse again from each of those for the **2-hop** set.

- **Auth & Tenancy →** every surface.
- **Config Store →** LLM Call, Evaluations-Text, Assessment. *(High blast radius: the same
  versioned `config_id + version` store backs production `/llm/call` AND eval runs — a schema
  or resolution change here ripples into live traffic.)*
- **Credentials →** LLM Call, Knowledge Base, Evaluations (all), Assessment, Response.
- **LLM Call →** Evaluations-Text (fast-evals path), Assessment (prefilter stages), Response.
- **Knowledge Base →** LLM Call (file_search), Evaluations-Text (RAG eval configs).
- **Evaluation Dataset →** Evaluations-Text/STT/TTS **and** Assessment (shared dataset table,
  distinguished by a `type` column: `text`/`stt`/`tts`/`assessment`).
- **Evaluations-Text →** Notifications.
- **Batch infra (`core/batch/`) →** Evaluations (all three), Assessment.
- **Job tracking →** LLM Call, Response.

## Known coupling hotspots

These are the non-obvious edges a spec most often forgets — check them explicitly:

1. **Config Store is shared by production and evals.** Touching `config`/`model_config`
   schema or `resolve_config_blob`/`resolve_evaluation_config` affects `/llm/call`,
   Evaluations-Text, and Assessment at once. *(evidence: `services/llm/jobs.py` imports
   `ConfigVersionCrud`; `evaluation_run.config_id` FK → `config.id`;
   `services/assessment/service.py` imports `resolve_evaluation_config`.)*
2. **One dataset table, four families.** `evaluation_dataset` is shared by text/STT/TTS evals
   and Assessment via a `type` discriminator. A dataset change touches all four.
3. **Knowledge Base feeds two consumers.** `knowledge_base_ids` is resolved into OpenAI
   file_search by both `/llm/call` and text eval configs.
4. **Langfuse is the system of record for text evals** (datasets, traces, LLM-judge scores),
   but NOT for STT/TTS (Postgres tables). Anything touching judge/scoring spans Langfuse +
   the eval scoring/merge code.
5. **Notifications fire only from text evals today** (STT/TTS do not). A new eval family that
   should notify must wire this explicitly.

## Provenance & flags

- Edges derived from: model `foreign_key=` declarations, cross-surface imports in
  `services/`+`crud/`, and `docs/architecture/{kaapi-llm-call,kaapi-evaluations,kaapi-knowledge-base}-ARCHITECTURE.md`.
- **Merged:** `model_evaluation` belongs with Fine-tuning (its FK is `fine_tuning_id`, not
  `evaluation_run_id`) — it is *not* part of the Evaluations surface.
- **Stale doc:** the evaluations arch doc says fast-evals are "not present on this branch," but
  `services/evaluations/fast.py` exists and imports `services.llm.providers`. Code wins —
  re-confirm during reconcile.
