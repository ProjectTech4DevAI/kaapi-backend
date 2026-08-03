# Module: Evaluations

Measures LLM config quality on golden datasets. Three families sharing one dataset/run schema: `text` (cosine + LLM-judge, OpenAI), `stt` (WER/CER, Gemini), `tts` (human annotation, Gemini). Batch-first (provider Batch APIs, not `/llm/call`), plus a synchronous `fast` text mode capped at `EVAL_FAST_MAX_UNIQUE_ROWS`.
Deep dive: `docs/architecture/kaapi-evaluations-ARCHITECTURE.md` (§3 data model, §5 text pipeline, §6 STT, §7 TTS, §8 batch infra, §9 cron, §12 design decisions).

All paths relative to `backend/app/`.

## Routes
- `api/routes/evaluations/dataset.py`, `api/routes/evaluations/evaluation.py` — text datasets + runs (v1, `/api/v1`)
- `api/routes/evaluations/evaluation_v2.py` — `POST /api/v2/evaluations`, replica of v1 run trigger + native ground-truth LLM judge (Langfuse-free); mounted under `settings.API_V2_STR`
- `api/routes/evaluations/dataset_v2.py` — `POST /api/v2/evaluations/datasets`, Langfuse-free dataset upload; stores only the original CSV in S3 and records `duplication_factor` as metadata (rows expanded ×factor at run time, not physically duplicated)
- `api/routes/evaluations/prompt_improvement_v2.py` — `POST /api/v2/evaluations/{evaluation_id}/improve-prompt`, prompt iteration off the three-metric judge results (requires an `is_judge_run` run); same body as v1, returns a recommendation of type `prompt`
- `api/routes/evaluations/iteration_v2.py` — `POST /api/v2/evaluations/iterations`, kicks off the automated eval → improve-prompt → eval loop (see Async); returns `202` with an `EvaluationIterationRunImmediatePublic` handle, final round-by-round report delivered to `callback_url`
- `api/routes/stt_evaluations/`, `api/routes/tts_evaluations/` — STT/TTS
- `api/routes/cron.py` — batch polling trigger

## Tables (SQLModel)
| Table | Model |
|---|---|
| `evaluation_dataset` (EvaluationDataset), `evaluation_run` (EvaluationRun) | `models/evaluation.py` |
| `stt_sample`, `stt_result` | `models/stt_evaluation.py` |
| `tts_result` | `models/tts_evaluation.py` |
| `batch_job` (BatchJob) | `models/batch_job.py` |
| `evaluation_iteration_run` (EvaluationIterationRun) | `models/evaluation_iteration.py` |

Key `EvaluationRun` JSONB fields: `score` (per-trace `scores` + `summary_scores`), `per_item_scores`, `cost` (per-stage), `unscoreable`. References `config_id`, `batch_job_id`.
v2 judge field on `EvaluationRun`: `is_judge_run` (bool marker gating native judging + Langfuse skip). All judge metrics are trace-only — per-row score + reasoning live in the `score_trace_url` trace unit (the native source of truth); there is no per-metric backup column. Judging is system-config only — always the fallback model (`gpt-5-mini`) + built-in prompts, no per-run config.

`EvaluationIterationRun` is a thin bookkeeping row only (`status`, `stop_reason`, `dataset_id`, `config_id`, `initial_config_version`, `callback_url`, `error_message`) — round-by-round state (`round_number`, `current_eval_run_id`, `current_improvement_job_id`, `history`, `best_*`, `consecutive_low_delta_rounds`) lives entirely in the LangGraph checkpoint keyed by `thread_id = str(id)`, not on this table. No FK to `EvaluationRun`/`Job`; those are referenced only inside the checkpoint state.

## Services / CRUD
- `services/evaluations/` — `evaluation.py`, `dataset.py` (`upload_dataset`; `use_langfuse=False` is the v2 Langfuse-free upload), `fast.py` (`validate_fast_evaluation_inputs` extracted for reuse by both the direct eval-start path and the iteration loop), `batch_job.py`, `validators.py`, `prompt_improvement.py`, `iteration.py` (`validate_and_start_evaluation_iteration`, `compute_round_scores`), `iteration_graph.py` (the LangGraph `StateGraph`: nodes, checkpointer)
- `services/stt_evaluations/`, `services/tts_evaluations/`
- `crud/evaluations/` — `core.py`, `batch.py`, `fast.py`, `judge.py` (`METRIC_REGISTRY` + combined judge call; `ground_truth`, `prompt`, and `knowledge_base` metrics, applied per-row by which required inputs the row carries), `score.py`, `embeddings.py`, `cost.py`, `langfuse.py`, `merge.py`, `processing.py`, `cron.py`, `iteration.py` (thin-row CRUD for the iteration loop)
- `core/batch/` — shared provider batch clients: `openai.py`, `gemini.py`, `anthropic.py`, `polling.py`, `operations.py`

## Async
- Provider batches polled by cron (`crud/evaluations/cron.py`); no long-lived Celery task per run.
- Fast text runs (v1 cosine + v2 judge) fan out `ceil(total_items / EVAL_FAST_CHUNK_SIZE)` `run_evaluation_fast_chunk` tasks (responses only), then a cron barrier (`dispatch_fast_evaluation_barriers`) enqueues one `run_evaluation_fast_aggregate` once every chunk has a `raw_output_url`. The aggregate merges chunks, then (v2) judges **every** row in that single task — so the judge pool is sized by its own `EVAL_JUDGE_CONCURRENCY` (not the response stage's `EVAL_FAST_API_CONCURRENCY`) to clear the max dataset (`EVAL_FAST_MAX_UNIQUE_ROWS` × `duplication_factor`) under the aggregate's `CELERY_TASK_SOFT_TIME_LIMIT`. No judge fan-out / second barrier.
- Prompt improvement is job-based with callback delivery: `POST /evaluations/{id}/improve-prompt` validates preconditions, enqueues a `Job` (`JobType.PROMPT_IMPROVEMENT`, `models/job.py`) run by Celery task `run_prompt_improvement` (`celery/tasks/job_execution.py`), and returns `202` with an `LLMJobImmediatePublic` handle. On finish the worker POSTs a single best-effort callback to the caller-supplied `callback_url` (SSRF-guarded via `validate_callback_url`): an `APIResponse[PromptImprovementJobPublic]` (`models/evaluation.py`) carrying the new `ConfigVersion` on success or `error_message` on failure. The `ConfigVersion` is persisted regardless of callback outcome. Celery redelivery of a `SUCCESS` job re-sends the callback without re-running the LLM.
- v2 prompt iteration reuses the same job/Celery/config-version machinery (`services/evaluations/prompt_improvement.py`) and branches on `run.is_judge_run`: the v2 route calls `start_prompt_improvement_job(..., require_judge_run=True)` (a non-judge run → `422 not_a_judge_run`), and the worker drafts from the three-metric judge trace (`_draft_improved_prompt(is_judge_run=True)`, using each metric's score + reasoning) and delivers a `PromptRecommendationJobPublic` callback carrying `recommendation_type` (`Literal["prompt"]`, `models/evaluation.py`; widens to a union when knowledge-base / model recommendations land). Non-judge (v1) runs keep the default `_draft_improved_prompt` brief + `PromptImprovementJobPublic` path unchanged.
- Evaluation iteration loop (`POST /evaluations/iterations`) chains fast-eval + v2 prompt improvement into a self-driving cycle via LangGraph (`services/evaluations/iteration_graph.py`): `start_eval_node` → `wait_eval_node` → (conditional) `start_improve_node` → `wait_improve_node` → loops back to `start_eval_node`, or → `finalize_node`. Stop-score = mean(`Adherence to Ground Truth`, `Adherence to Prompt`) from `EvaluationRun.score["summary_scores"]` (`compute_round_scores`); `Adherence to Knowledge Base` is recorded per round for visibility only, never gates stopping. Stops on 3 consecutive rounds with `<EVAL_ITERATION_CEILING_DELTA_THRESHOLD` (0.05) improvement (`ceiling_reached`), a `max_rounds` cap (`max_rounds_reached`), or a hard round failure (`round_failed`). No orchestrator polls a provider directly: each `wait_*_node` reads `EvaluationRun.status`/`Job.status` (both already maintained by the existing fast-eval and prompt-improvement machinery) and calls LangGraph's `interrupt()` if not yet terminal — a Postgres checkpointer (`get_evaluation_iteration_checkpointer`) persists state across the pause. Resumption is driven by the existing `/cron/evaluations` tick, not a new scheduler: `dispatch_pending_evaluation_iteration_resumes` (`crud/evaluations/cron.py`) re-dispatches `run_evaluation_iteration_graph_step` (Celery, `celery/tasks/job_execution.py`, priority 6) for every `EvaluationIterationRun` with `status=processing`. Every node opens/closes its own DB session — no session is held open across an `interrupt()`. `finalize_node` persists the thin row's terminal `status`/`stop_reason` and delivers the round history + best round as an `EvaluationIterationReportPublic` callback, same best-effort delivery convention as prompt improvement above.

## External
- OpenAI Batch + embeddings, Gemini Batch, Langfuse (per-trace scores + summary), object storage (datasets/audio), kaapi-frontend console (results + annotation).
- LangGraph (`langgraph`, `langgraph-checkpoint-postgres`) for the iteration loop — its `PostgresSaver` checkpointer connects via its own `psycopg` (v3) connection pool against the same database, separate from the app's SQLAlchemy engine; owns its own tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`), not Alembic-managed.

## Gotchas
- Eval traffic deliberately bypasses `/llm/call` (separate code path from production).
- Scores sync to Langfuse; durable per-row maps on `evaluation_run` are the resync source of truth.
- v2 (`is_judge_run`) is fully Kaapi-native: no Langfuse dataset/trace/score sync; one combined `responses.create` per row scores every applicable metric, structured via `crud/evaluations/judge.py` `METRIC_REGISTRY`. Live metrics (all trace-only — score + reasoning land in `score_trace_url`, no backup columns): `ground_truth`, `prompt` (obedience to the assistant's configured instructions; applies only when the run resolves a config prompt), and `knowledge_base` (groundedness; judges the file_search chunks captured during response generation, applies only to rows that retrieved chunks). Every run judges the whole registry; `_applicable_metrics` drops, per row, any metric whose required inputs that row cannot supply — including `prompt` when the run resolved no config prompt (it is passed as `""` for every row). Each metric's prompt fragment states its own CONSIDER/IGNORE input scope, so `_INPUT_LABELS` and those fragments must be edited together. v1 path is unchanged.
