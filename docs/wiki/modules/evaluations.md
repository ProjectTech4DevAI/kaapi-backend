# Module: Evaluations

Measures LLM config quality on golden datasets. Three families sharing one dataset/run schema: `text` (cosine + LLM-judge, OpenAI), `stt` (WER/CER, Gemini), `tts` (human annotation, Gemini). Batch-first (provider Batch APIs, not `/llm/call`), plus a synchronous `fast` text mode capped at `EVAL_FAST_MAX_UNIQUE_ROWS`.
Deep dive: `docs/architecture/kaapi-evaluations-ARCHITECTURE.md` (§3 data model, §5 text pipeline, §6 STT, §7 TTS, §8 batch infra, §9 cron, §12 design decisions).

All paths relative to `backend/app/`.

## Routes
- `api/routes/evaluations/dataset.py`, `api/routes/evaluations/evaluation.py` — text datasets + runs
- `api/routes/stt_evaluations/`, `api/routes/tts_evaluations/` — STT/TTS
- `api/routes/cron.py` — batch polling trigger

## Tables (SQLModel)
| Table | Model |
|---|---|
| `evaluation_dataset` (EvaluationDataset), `evaluation_run` (EvaluationRun) | `models/evaluation.py` |
| `stt_sample`, `stt_result` | `models/stt_evaluation.py` |
| `tts_result` | `models/tts_evaluation.py` |
| `batch_job` (BatchJob) | `models/batch_job.py` |

Key `EvaluationRun` JSONB fields: `score` (per-trace `scores` + `summary_scores`), `per_item_scores`, `cost` (per-stage), `unscoreable`. References `config_id`, `batch_job_id`.

## Services / CRUD
- `services/evaluations/` — `evaluation.py`, `dataset.py`, `fast.py`, `batch_job.py`, `validators.py`, `prompt_improvement.py`
- `services/stt_evaluations/`, `services/tts_evaluations/`
- `crud/evaluations/` — `core.py`, `batch.py`, `fast.py`, `score.py`, `embeddings.py`, `cost.py`, `langfuse.py`, `merge.py`, `processing.py`, `cron.py`
- `core/batch/` — shared provider batch clients: `openai.py`, `gemini.py`, `anthropic.py`, `polling.py`, `operations.py`

## Async
- Provider batches polled by cron (`crud/evaluations/cron.py`); no long-lived Celery task per run.
- Prompt improvement is job-based with callback delivery: `POST /evaluations/{id}/improve-prompt` validates preconditions, enqueues a `Job` (`JobType.PROMPT_IMPROVEMENT`, `models/job.py`) run by Celery task `run_prompt_improvement` (`celery/tasks/job_execution.py`), and returns `202` with an `LLMJobImmediatePublic` handle. On finish the worker POSTs a single best-effort callback to the caller-supplied `callback_url` (SSRF-guarded via `validate_callback_url`): an `APIResponse[PromptImprovementJobPublic]` (`models/evaluation.py`) carrying the new `ConfigVersion` on success or `error_message` on failure. The `ConfigVersion` is persisted regardless of callback outcome. Celery redelivery of a `SUCCESS` job re-sends the callback without re-running the LLM.

## External
- OpenAI Batch + embeddings, Gemini Batch, Langfuse (per-trace scores + summary), object storage (datasets/audio), kaapi-frontend console (results + annotation).

## Gotchas
- Eval traffic deliberately bypasses `/llm/call` (separate code path from production).
- Scores sync to Langfuse; durable per-row maps on `evaluation_run` are the resync source of truth.
