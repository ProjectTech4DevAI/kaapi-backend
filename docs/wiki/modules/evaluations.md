# Module: Evaluations

Measures LLM config quality on golden datasets. Three families sharing one dataset/run schema: `text` (cosine + LLM-judge, OpenAI), `stt` (WER/CER, Gemini), `tts` (human annotation, Gemini). Batch-first (provider Batch APIs, not `/llm/call`), plus a synchronous `fast` text mode capped at `EVAL_FAST_MAX_UNIQUE_ROWS`.
Deep dive: `docs/architecture/kaapi-evaluations-ARCHITECTURE.md` (§3 data model, §5 text pipeline, §6 STT, §7 TTS, §8 batch infra, §9 cron, §12 design decisions).

All paths relative to `backend/app/`.

## Routes
- `api/routes/evaluations/dataset.py`, `api/routes/evaluations/evaluation.py` — text datasets + runs (v1, `/api/v1`)
- `api/routes/evaluations/evaluation_v2.py` — `POST /api/v2/evaluations`, replica of v1 run trigger + native ground-truth LLM judge (Langfuse-free); mounted under `settings.API_V2_STR`
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
v2 judge fields on `EvaluationRun`: `is_judge_run` (bool marker gating native judging + Langfuse skip), `per_item_ground_truth` (JSONB `{ref: score}`, Kaapi-native, not synced to Langfuse). Judging is system-config only — always the fallback model (`gpt-5-mini`) + built-in prompt, no per-run config.

## Services / CRUD
- `services/evaluations/` — `evaluation.py`, `dataset.py`, `fast.py`, `judge.py` (v2 judged run trigger), `batch_job.py`, `validators.py`, `prompt_improvement.py`
- `services/stt_evaluations/`, `services/tts_evaluations/`
- `crud/evaluations/` — `core.py`, `batch.py`, `fast.py`, `judge.py` (metric registry + combined judge call), `score.py`, `embeddings.py`, `cost.py`, `langfuse.py`, `merge.py`, `processing.py`, `cron.py`
- `core/batch/` — shared provider batch clients: `openai.py`, `gemini.py`, `anthropic.py`, `polling.py`, `operations.py`

## Async
- Provider batches polled by cron (`crud/evaluations/cron.py`); no long-lived Celery task per run.

## External
- OpenAI Batch + embeddings, Gemini Batch, Langfuse (per-trace scores + summary), object storage (datasets/audio), kaapi-frontend console (results + annotation).

## Gotchas
- Eval traffic deliberately bypasses `/llm/call` (separate code path from production).
- Scores sync to Langfuse; durable per-row maps on `evaluation_run` are the resync source of truth.
- v2 (`is_judge_run`) is fully Kaapi-native: no Langfuse dataset/trace/score sync; the native ground-truth judge is one combined `responses.create` per row, structured via `crud/evaluations/judge.py` `METRIC_REGISTRY` so knowledge_base/prompt metrics + verdict slot in later. v1 path is unchanged.
