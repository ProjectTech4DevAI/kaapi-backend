# Domain Map

Entities and their edges. Use for blast-radius analysis: when a feature changes an entity, walk its `consumed by` edges 1-hop and 2-hop, and confirm scope for every surface the spec does not mention.

FK edges below are generated from `foreign_key=` declarations in `backend/app/models/`. Logical edges (no FK) are marked `(logical)`.

## Tenancy spine

Every multi-tenant table carries `organization_id` + `project_id`.

```
Organization ─< Project ─< (almost everything below)
User ─< UserProject >─ Project        # membership + role
APIKey → Organization, Project, User  # programmatic access
```

## Entities and edges

| Entity (table) | Model file | Belongs to / references | Consumed by |
|---|---|---|---|
| Organization | organization.py | — | Project, APIKey, Credential, FeatureFlag, and every tenant-scoped table |
| Project | project.py | Organization | nearly all tables; unit of permissioning |
| User | user.py | — | UserProject, APIKey, Notification |
| APIKey | api_key.py | Organization, Project, User | auth dependency on every API route (logical) |
| Credential | credentials.py | Organization, Project | provider clients: OpenAI/Gemini/Anthropic calls (logical) |
| Config | config/config.py | Project | ConfigVersion; LLM call path; Assessment; EvaluationRun (`config_id`) |
| ConfigVersion | config/version.py | Config | resolved by `LLMCallConfig` saved references (logical) |
| LlmCall | llm/request.py | Job, LlmChain, Org, Project | Langfuse traces (logical); analytics |
| LlmChain | llm/request.py | Org, Project | LlmCall |
| Job | job.py | Project | LlmCall; Celery job execution (logical) |
| BatchJob | batch_job.py | Org, Project | EvaluationRun, Assessment; batch polling cron (logical) |
| EvaluationDataset | evaluation.py | Org, Project, Language | EvaluationRun, STTSample (via stt_evaluation), Assessment |
| EvaluationRun | evaluation.py | Dataset, Config, BatchJob, Org, Project, Language | STTResult, TTSResult; Langfuse scores (logical); console UI (logical) |
| STTSample / STTResult | stt_evaluation.py | Dataset, Run, File, Language | human annotation UI (logical) |
| TTSResult | tts_evaluation.py | Run, Org, Project | human annotation UI (logical) |
| Assessment / AssessmentRun | assessment.py | Config, Dataset, BatchJob, Org, Project | console UI (logical) |
| Document | document.py | Project, Document (parent) | DocumentCollection, DocTransformationJob, FineTuning, ModelEvaluation |
| Collection | collection.py | Project | DocumentCollection, CollectionJob; provider vector stores (logical) |
| DocumentCollection | document_collection.py | Document, Collection | RAG lookups (logical) |
| CollectionJob | collection_job.py | Collection, Project | async collection create/delete (logical) |
| DocTransformationJob | doc_transformation_job.py | Document | doctransform service (logical) |
| File | file.py | Org, Project | STT samples; object storage keys (logical) |
| FineTuning | fine_tuning.py | Document, Org, Project | ModelEvaluation |
| ModelEvaluation | model_evaluation.py | FineTuning, Document, Org, Project | console UI (logical) |
| Assistant | assistants.py | Org, Project | OpenAI assistants API (logical) |
| OpenAIConversation | openai_conversation.py | Org, Project | response service (logical) |
| OpenAIThread | threads.py | — | thread results read path (logical) |
| Language | language.py (`global.languages`) | — | EvaluationDataset, EvaluationRun, STT |
| FeatureFlag | feature_flag.py | Org, Project | route gating (logical) |
| Notification | notification.py | Project, User | notification service |

## External consumers (always check in blast radius)

- **Langfuse** — every LLM call and evaluation run writes traces/scores. A change to run scoring or trace shape ripples here.
- **kaapi-frontend console** — reads run results, annotation queues, config CRUD. A response-shape change ripples here.
- **Provider Batch APIs** (OpenAI, Gemini, Anthropic in `core/batch/`) — eval/assessment payload shape changes ripple here.
- **Object storage** (`core/cloud/storage.py`) — files, dataset artifacts.

## Blast-radius procedure

1. Name the primary entity(ies) the feature changes, in this file's vocabulary.
2. Collect 1-hop and 2-hop `consumed by` surfaces (tables AND logical/external consumers).
3. For every surface the spec does not address: ask the user — in scope / deferred / out of scope. Never silently include or exclude.
4. Record the decisions in the SRD (Assumptions out-of-scope bullets or a small table).
