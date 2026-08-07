# Module: Responses & Assistants

OpenAI Responses API integration: conversations, threads, assistants, async response jobs. No deep-dive doc yet.

All paths relative to `backend/app/`.

## Routes
- `api/routes/responses.py` — response creation
- `api/routes/openai_conversation.py` — conversation management
- `api/routes/threads.py` — thread results
- `api/routes/assistants.py` — assistant CRUD

## Tables (SQLModel)
| Table | Model |
|---|---|
| `openai_conversation` (OpenAIConversation) | `models/openai_conversation.py` |
| `openai_thread` (OpenAIThread) | `models/threads.py` |
| `assistant` (Assistant) | `models/assistants.py` |

Related schemas: `models/response.py`, `models/message.py`.

## Services / CRUD
- `services/response/` — `response.py`, `jobs.py`, `callbacks.py` (job execution + webhooks)
- `crud/openai_conversation.py`, `crud/assistants.py`, `crud/thread_results.py`

## Async
- Response jobs run via Celery (`services/response/jobs.py` + `celery/tasks/job_execution.py`), callback on completion.

## External
- OpenAI Responses/Assistants APIs, Langfuse tracing, client callback URLs.
