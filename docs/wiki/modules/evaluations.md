# Module: Evaluations

Measures LLM config quality on golden datasets. Three families sharing one dataset/run schema: `text` (cosine + LLM-judge, OpenAI), `stt` (WER/CER, Gemini), `tts` (human annotation, Gemini). Batch-first (provider Batch APIs, not `/llm/call`), plus a synchronous `fast` text mode capped at `EVAL_FAST_MAX_UNIQUE_ROWS`.
Deep dive: `docs/architecture/kaapi-evaluations-ARCHITECTURE.md` (§3 data model, §5 text pipeline, §6 STT, §7 TTS, §8 batch infra, §9 cron, §12 design decisions).

All paths relative to `backend/app/`.

## Routes
- `api/routes/evaluations/dataset.py`, `api/routes/evaluations/evaluation.py` — text datasets + runs (v1, `/api/v1`)
- `api/routes/evaluations/evaluation_v2.py` — `POST /api/v2/evaluations`, replica of v1 run trigger + native ground-truth LLM judge (Langfuse-free); mounted under `settings.API_V2_STR`
- `api/routes/evaluations/dataset_v2.py` — `POST /api/v2/evaluations/datasets`, Langfuse-free dataset upload; stores only the original CSV in S3 and records `duplication_factor` as metadata (rows expanded ×factor at run time, not physically duplicated)
- `api/routes/evaluations/prompt_improvement_v2.py` — `POST /api/v2/evaluations/{evaluation_id}/improve-prompt`, prompt iteration off the three-metric judge results (requires an `is_judge_run` run); same body as v1, returns a recommendation of type `prompt`
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
v2 judge field on `EvaluationRun`: `is_judge_run` (bool marker gating native judging + Langfuse skip). All judge metrics are trace-only — per-row score + reasoning live in the `score_trace_url` trace unit (the native source of truth); there is no per-metric backup column. Judging is system-config only — always the fallback model (`gpt-5-mini`) + built-in prompts, no per-run config. Judge metrics score on an **integer 0–5 stepped scale** (the LLM returns integers 0–5; `crud/evaluations/judge.py::_parse_metric_score` enforces it) with English-only reasoning; scores are stored raw (0–5), so the API structure is unchanged but the value range is 0–5 (cosine on the v1 path stays 0–1). Each numeric judge-metric trace score also carries a `verdict` band (`crud/evaluations/score.py`: `VerdictEnum` + `verdict_from_score`, 0–5 cutoffs 2/4 → 0–1 Needs Improvement, 2–3 Needs Refinement, 4–5 Good), set in the `crud/evaluations/fast.py` trace-build loop; cosine and unscoreable/`N/A` entries carry none.
`score.overall` (`OverallSummary`, `crud/evaluations/score.py`): run-level weighted rollup for judge runs, computed by `compute_overall_summary` from each metric's `avg` + `METRIC_REGISTRY` weight (renormalized over metrics that actually scored ≥1 row) — `overall_score`, `verdict`, per-metric `breakdown` (score/weight/delta/verdict), plus `ai_summary` (best-effort natural-language note, `crud/evaluations/summary.py`, Anthropic call via the platform-owned `settings.ANTHROPIC_API_KEY` (same key as prompt improvement, no per-project credential), model = `settings.EVAL_SUMMARY_MODEL`; `None` on any failure, never fails the run). Persisted on `run_fast_evaluation`'s final `EvaluationRun.score` write and re-attached verbatim by `services/evaluations/evaluation.py::get_evaluation_with_scores` on every cache/resync path (trace merging never recomputes it).

## Services / CRUD
- `services/evaluations/` — `evaluation.py`, `dataset.py` (`upload_dataset`; `use_langfuse=False` is the v2 Langfuse-free upload), `fast.py`, `batch_job.py`, `validators.py`, `prompt_improvement.py`
- `services/stt_evaluations/`, `services/tts_evaluations/`
- `crud/evaluations/` — `core.py`, `batch.py`, `fast.py`, `judge.py` (`METRIC_REGISTRY` + combined judge call; `ground_truth`, `prompt`, and `knowledge_base` metrics, applied per-row by which required inputs the row carries, each spec carrying a `weight` for the overall rollup), `score.py` (`VerdictEnum`/`verdict_from_score`, `OverallSummary`/`compute_overall_summary`), `summary.py` (`generate_run_ai_summary` — one-shot Anthropic `messages.create` call via `ClaudeProvider`, structured JSON output; prompt carries every trace's raw per-question scores + judge rationale, golden/generated answers, and the evaluated config, and returns a severity-ranked diagnostic note, not just a qualitative band summary), `embeddings.py`, `cost.py`, `langfuse.py`, `merge.py`, `processing.py`, `cron.py`
- `core/batch/` — shared provider batch clients: `openai.py`, `gemini.py`, `anthropic.py`, `polling.py`, `operations.py`

## Async
- Provider batches polled by cron (`crud/evaluations/cron.py`); no long-lived Celery task per run.
- Fast text runs (v1 cosine + v2 judge) fan out `ceil(total_items / EVAL_FAST_CHUNK_SIZE)` `run_evaluation_fast_chunk` tasks (responses only), then a cron barrier (`dispatch_fast_evaluation_barriers`) enqueues one `run_evaluation_fast_aggregate` once every chunk has a `raw_output_url`. The aggregate merges chunks, then (v2) judges **every** row in that single task — so the judge pool is sized by its own `EVAL_JUDGE_CONCURRENCY` (not the response stage's `EVAL_FAST_API_CONCURRENCY`) to clear the max dataset (`EVAL_FAST_MAX_UNIQUE_ROWS` × `duplication_factor`) under the aggregate's `CELERY_TASK_SOFT_TIME_LIMIT`. No judge fan-out / second barrier.
- Prompt improvement is job-based with callback delivery: `POST /evaluations/{id}/improve-prompt` validates preconditions, enqueues a `Job` (`JobType.PROMPT_IMPROVEMENT`, `models/job.py`) run by Celery task `run_prompt_improvement` (`celery/tasks/job_execution.py`), and returns `202` with an `LLMJobImmediatePublic` handle. On finish the worker POSTs a single best-effort callback to the caller-supplied `callback_url` (SSRF-guarded via `validate_callback_url`): an `APIResponse[PromptImprovementJobPublic]` (`models/evaluation.py`) carrying the new `ConfigVersion` on success or `error_message` on failure. The `ConfigVersion` is persisted regardless of callback outcome. Celery redelivery of a `SUCCESS` job re-sends the callback without re-running the LLM.
- v2 prompt iteration reuses the same job/Celery/config-version machinery (`services/evaluations/prompt_improvement.py`) and branches on `run.is_judge_run`: the v2 route calls `start_prompt_improvement_job(..., require_judge_run=True)` (a non-judge run → `422 not_a_judge_run`), and the worker drafts from the three-metric judge trace (`_draft_improved_prompt(is_judge_run=True)`, using each metric's score + reasoning) and delivers a `PromptRecommendationJobPublic` callback carrying `recommendation_type` (`Literal["prompt"]`, `models/evaluation.py`; widens to a union when knowledge-base / model recommendations land). Non-judge (v1) runs keep the default `_draft_improved_prompt` brief + `PromptImprovementJobPublic` path unchanged.

## External
- OpenAI Batch + embeddings, Gemini Batch, Langfuse (per-trace scores + summary), object storage (datasets/audio), kaapi-frontend console (results + annotation).

## Gotchas
- Eval traffic deliberately bypasses `/llm/call` (separate code path from production).
- Scores sync to Langfuse; durable per-row maps on `evaluation_run` are the resync source of truth.
- v2 (`is_judge_run`) is fully Kaapi-native: no Langfuse dataset/trace/score sync; one combined `responses.create` per row scores every applicable metric, structured via `crud/evaluations/judge.py` `METRIC_REGISTRY`. Live metrics (all trace-only — score + reasoning land in `score_trace_url`, no backup columns): `ground_truth`, `prompt` (obedience to the assistant's configured instructions; applies only when the run resolves a config prompt), and `knowledge_base` (groundedness; judges the file_search chunks captured during response generation, applies only to rows that retrieved chunks). Every run judges the whole registry; `_applicable_metrics` drops, per row, any metric whose required inputs that row cannot supply — including `prompt` when the run resolved no config prompt (it is passed as `""` for every row). Each metric's prompt fragment states its own CONSIDER/IGNORE input scope, so `_INPUT_LABELS` and those fragments must be edited together. v1 path is unchanged.
