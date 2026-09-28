# Plan: share `/llm/call`'s code with evaluation, and run eval rows through guardrails

**Branch:** `evaluation-guardrail`
**Status:** Phases 1, 2 and 3 done (uncommitted). Clean-DB suite green, `/pr-review` run, and
end-to-end exercised against a dev project (2026-09-27) — see "Verification run" below. Review nits
fixed and `TestRetryOpenAICall` dropped on the user's call. Still open: a real *blocking*
guardrail run (local guardrails service is down), Verification item 5 (Sentry/OTel spans, never
run), and whether the untracked `docs/plans/` ships in the PR.
**Scope decided with the user.** Do not re-litigate the choices in "Scope" — they are answers, not suggestions.

## How to use this file

This is a cold-start brief. Read, in order:

1. This file, top to bottom.
2. `docs/wiki/modules/guardrails.md` — §4 failure-semantics matrix, §6 inline hooks, §7 metadata, §9 invariants.
3. `docs/wiki/modules/evaluations.md` and `docs/wiki/modules/llm-call.md`.

Then work the phases in order (Phase 2 depends on Phase 1). Per root `CLAUDE.md`: layer edits go
through the `senior-engineer` subagent, tests through `test-writer`, then `/pr-review` on the full
diff before committing.

---

## In plain terms

Evaluation has its own copy of "call the model". We delete that copy and have it call the same
code production uses. Because guardrails already live inside that shared code, evaluation gets
guardrails for free the moment it switches over.

Three steps:

1. **Make the shared call usable by evaluation.** It currently insists on writing an audit row
   and reporting itself to production monitoring. Add a switch that turns both off, and give it
   a way to say "guardrails stopped this" instead of just "something went wrong".
2. **Point evaluation at it.** Swap evaluation's private model call for the shared one, with the
   switch turned off. Evaluation keeps its own result format, scoring and storage — only the
   invocation changes.
3. **Decide what a guardrail-stopped row means for a score.** It is not a crash, so it must not
   count as a failure; it just has nothing to score. Make both scoring paths treat it that way
   and show the reason.

Scoring, judging, embeddings, batch runs, the summary and prompt improvement are untouched.

---

## Context

Today `crud/evaluations/fast.py::_responses_call_for_item` calls
`openai_client.responses.create(...)` directly, with its own param assembly, its own retry
wrapper and its own result shape. `/llm/call` runs a completely separate path through
`services/llm/jobs.py::execute_llm_call` → `providers/*.execute`.

Two costs:

1. **Two functions to maintain.** Anything added to the `/llm/call` pipeline has to be
   re-implemented in eval or it silently doesn't apply there.
   `docs/architecture/kaapi-evaluations-ARCHITECTURE.md` §12.1 records this as an accepted
   downside — "an eval does not run the same code that production will run". §12.2 records
   reusing `/llm/call` as the intended fix; it was never built.
2. **Evals don't run guardrails.** `resolve_evaluation_config` returns a full `ConfigBlob`
   carrying `input_guardrails` / `output_guardrails`, and evaluation reads only
   `completion.params`. Evaluating a guardrailed config measures the bare model, not what
   production serves. Verified: zero `guardrail` references anywhere under
   `services/evaluations/`, `crud/evaluations/`, `api/routes/evaluations/`.

---

## Scope

- **Fast mode only** (`run_mode=fast`, which is also all of v2). The batch path stays on provider
  Batch APIs — a batch submission is a JSONL upload and structurally cannot route through a
  per-call function. Batch runs keep having no guardrails.
- **A guardrail-blocked row is unscoreable, not failed.** It must not count toward
  `EVAL_FAST_FAILURE_THRESHOLD` in **either** stage — the v1 embedding stage has a second
  threshold check that would otherwise catch these rows (see Phase 3).
- **Latency/throughput tuning is deferred.** Ship, measure, tune. See Risks.
- **Assumption to confirm at review:** a **rephrased** row (guardrails answered directly, no LLM
  call) is scored normally, because that canned text is exactly what production would have
  returned. It will drag scores down when compared against ground truth. Treating rephrase as
  unscoreable like a block is a one-line change either way.

Unchanged and explicitly out of scope: embeddings (v1 cosine), the LLM judge, the AI summary and
prompt improvement all keep their direct SDK calls and never get guardrails.

---

## Phase 1 — make `execute_llm_call` usable without persistence or telemetry

**DONE.** Built as specced, with two corrections found while reading the code — keep both in mind for
Phase 2:

- `outcome.error` is **not** always a content block: `run_guardrails_validation` also sets it when
  the guardrails service fail-closes on HTTP 401/403/422. Tagging that `"blocked"` would turn broken
  guardrail credentials into a "completed" eval where every row is silently unscoreable. The auth
  dict now carries `auth_error: True` and `GuardrailsOutcome.blocked` is the discriminator, so
  `guardrail_outcome` stays `None` on an auth failure and the row fails normally. **Phase 2 must key
  unscoreable off `guardrail_outcome`, never off `error`.**
- The proxy-branch output block carries `usage=proxy_usage`, not `response.usage` — `response` is
  not bound until after that branch returns, so the plan's original text would have raised
  `UnboundLocalError` into the outer handler and lost the block entirely.

Tests: `app/tests/services/llm/test_execute_llm_call.py` (new, 13), plus 6 in `test_guardrails.py`
and 3 in `test_mappers.py`. `uv run pytest app/tests/services/llm/ -q` → 528 passed, 6 failed; those
6 are pre-existing local-DB drift (`column "metadata" of relation "llm_call" does not exist`,
migration 083 unapplied) and fail identically with the change stashed.

Carry into Phase 2: `test_fast_judge.py::TestFileSearchIncludeParam` covers eval's current manual
`include` injection and must be updated when `run_response_chunk` stops setting it by hand.

### Original spec


### `backend/app/services/llm/chain/types.py`

Add `guardrail_outcome: Literal["blocked", "rephrased"] | None = None` to `BlockResult`.
`BlockResult.error` is a bare string today, so a caller cannot tell a guardrail block from a
provider failure — eval needs that distinction to decide unscoreable vs failed.

### `backend/app/services/llm/jobs.py::execute_llm_call`

- Add `record_call: bool = True` (keyword-only, like every other param). One flag covering the
  `LlmCall` row **and** telemetry: "this call is not production traffic — keep it out of the
  audit table, the AI spans and the LLM metrics."
- Skip the three `LlmCall` **creators** when it is `False`, leaving `llm_call_id = None`:
  - `create_llm_call` on the main path (inside the `llm.create_call_record` span)
  - `create_llm_call` in the proxy branch
  - `save_rephrase_guardrail_call` in the rephrase short-circuit
- Add the missing `if llm_call_id:` guard around the proxy branch's `update_llm_call_response`.
  Every other downstream write is already gated on `llm_call_id` (STT S3 upload +
  `update_llm_call_input`, TTS S3 upload, the main-path `update_llm_call_response`, and
  `persist_output_guardrail_result`, which early-returns on a falsy id). The proxy branch's
  update is the one unguarded site and would raise once the id can be `None`.
- Set `guardrail_outcome` at the three guardrail return sites: the input hard block, the rephrase
  short-circuit, and the output hard block. The **output** guardrail has two call sites (proxy
  branch + main provider path) — both need it, same as every other output-guardrail change
  (`guardrails.md` §10).
- At the two **output**-block sites, also set `usage=response.usage` on the returned
  `BlockResult`. The LLM was called and the tokens were spent; today the block returns a bare
  `BlockResult(error=...)` and the cost vanishes. Eval sums usage per chunk, so this would
  under-report run cost.
- Skip `record_llm_call_started` / `record_llm_call_finished` when `record_call` is `False`.
- Select the tracer per call: `_tracer = tracer if record_call else trace.NoOpTracer()`, and use
  `_tracer` for the eight `start_as_current_span` blocks in this function. **Verify first** that
  `NoOpTracer().start_as_current_span(...)` works as a drop-in context manager and that the
  yielded span tolerates `set_attribute` / `set_status` / `record_exception` (it yields
  `INVALID_SPAN`, a `NonRecordingSpan`, which should no-op all three).

Why silence telemetry rather than "spans are cheap": the main span is tagged
`sentry.op = "gen_ai.chat"` and `record_llm_call_*` emits per-provider/model/project metrics. A
100-row eval of a broken config would otherwise land in the production Sentry AI dashboards and
error-rate metrics — exactly the prod/eval mixing arch §12.2 flagged.

**Still emitted with `record_call=False`,** and out of scope to silence: the three spans from
`services/llm/guardrails.py`'s own module tracer, and the generic HTTP client spans from the
global `HTTPXClientInstrumentor` / `RequestsInstrumentor` (eval's direct SDK calls already emit
those today, so this is not a regression).

**Langfuse** needs no code change: pass `langfuse_credentials=None` and
`core/langfuse/langfuse.py::observe_llm_execution` returns the undecorated function. Eval also
skips the credential DB read entirely.

Net diff is small because the `llm_call_id` gating already exists. Do **not** restructure
`execute_llm_call` into extracted phases — it is the hottest path in the service and
`ChainBlock.execute` is a second live caller.

### `backend/app/services/llm/mappers.py::transform_kaapi_config_to_native`

OpenAI branch only — when the mapped params carry a `file_search` tool, set
`mapped_params["include"] = ["file_search_call.results"]`.

Eval sets this by hand today in `run_response_chunk` because the v2 `knowledge_base` judge metric
scores the retrieved chunks. Once generation goes through `execute_llm_call`, eval has no place
to inject it — an undeclared key on `TextLLMParams` is dropped silently at validation
(`llm-call.md`, "Key pydantic/SQLModel schemas").

Put it in `transform_kaapi_config_to_native`, **not** in `map_kaapi_to_openai_params`. The mapper
has five other callers — batch eval JSONL (`crud/evaluations/batch.py`), the judge
(`crud/evaluations/judge.py`), and two assessment batch paths (`crud/assessment/batch.py`,
`services/assessment/api/batch.py`) — and three of those build **Batch API** bodies. The
transform is only reached from `execute_llm_call`, so `/llm/call` and eval fast get it and
nothing else changes.

---

## Phase 2 — eval's generation stage calls `execute_llm_call`

**DONE.** Built as specced. Deviations and things Phase 3 inherits:

- **BLOCKER: Phase 3 must land in the same PR.** For **v1 (cosine)** a blocked row now has
  `failed=False, generated_output=""`, so it enters `_stage2_embeddings`'s `embed_candidates`, gets
  `build_embedding_failure`, and counts toward that stage's own threshold check — a guardrailed v1
  run blocking more than `EVAL_FAST_FAILURE_THRESHOLD` (0.5) of rows will **fail**, the exact
  behaviour this plan rejects. v2 is already safe (`classify_empty_side` / `select_judgeable_rows`
  skip empty output). `_stage2_embeddings` was deliberately not touched here.
- The retry had to become **result-based**, not exception-based: `execute_llm_call` converts
  provider transients into `BlockResult(error=...)` and never raises. New
  `retry.py::retry_llm_call` uses `retry_if_result`. Its `retry_error_callback` is load-bearing —
  `reraise` does **not** apply to `retry_if_result`, so without it an exhausted retry raises
  `RetryError` through `_run_in_pool`'s `future.result()`, killing the whole chunk; the cron healer
  then re-enqueues it and the provider is re-charged for every row already generated.
  `retry_openai_call` stays for embeddings and the judge.
- `QueryParams` **and** `job_id` are built inside the retried function, not per row.
  `execute_llm_call` mutates `query.input.content.value` twice in place (template interpolation,
  then the guardrails' `safe_text`), so a retried attempt reusing the object would double-apply the
  template and resend already-sanitised text.
- A **blocked** `BlockResult` has `response=None` (both the input block and the output block), so
  every read of `result.response.*` is guarded — `_response_text()` and
  `... if result.response else None`. Unguarded, the `AttributeError` would have been swallowed by
  the worker's `except Exception` into `failed=True`, silently defeating the whole change.
- The mapping checks `guardrail_outcome` **before** `error`: a rephrased row carries no error and a
  real response, so checked later it reads as a plain success.
- `openai_client` was dropped from `_resolve_config_and_clients`' return tuple rather than threaded
  through unused — it had exactly one consumer, `run_response_chunk`. Embeddings and the judge build
  their own clients in the aggregate.
- The `batch_job` config's `"model"` is read off `config_blob.completion.params` with the existing
  dict-or-typed-model idiom from `core.py::resolve_model_from_config`, avoiding a second config
  resolution round trip.
- **Convention deviations to flag at `/pr-review`:** `crud/evaluations/{fast,retry}.py` now import
  from `app/services/llm/` (`.claude/conventions/crud.md` forbids crud → service), verified free of
  import cycles; and `_llm_call_for_item` catches bare `except Exception` (crud.md wants concrete
  types) so one bad row cannot abort the pool.
- Tests: `test-writer` added 36, deleted 5, rewrote 5. Suite delta against the pre-change baseline is
  clean — no new assertion failures; the 46 failures are the same pre-existing local DB drift
  (`sqlalchemy.exc.ProgrammingError`). A clean-DB run is still owed before the PR.

### `backend/app/services/evaluations/fast.py`

- `_resolve_config_and_clients` currently narrows to
  `TextLLMParams.model_validate(config_blob.completion.params)` and throws the blob away. Return
  the `ConfigBlob` instead (the `OpenAI` client is still needed for embeddings + judge, the
  Langfuse client for v1 datasets).
- `execute_fast_evaluation_chunk` passes the blob down to `run_response_chunk`.
- `validate_fast_evaluation_inputs`: **reject a config whose `prompt_template` is set but does not
  contain `{{input}}`.** See Risks — this is the one silent-breakage path in the change.

### `backend/app/crud/evaluations/fast.py`

- `run_response_chunk(..., config: TextLLMParams, ...)` → takes `config_blob: ConfigBlob` and the
  ids it needs off `eval_run`. Drop the `map_kaapi_to_openai_params` call and the manual
  `base_params["include"]` line — `execute_llm_call` owns both now.
- Replace `_responses_call_for_item` with `_llm_call_for_item`, still the `_run_in_pool` worker at
  `EVAL_FAST_API_CONCURRENCY`:

```python
result = execute_llm_call(
    config=LLMCallConfig(id=eval_run.config_id, version=eval_run.config_version),
    query=QueryParams(input=TextInput(content=TextContent(value=question))),
    job_id=uuid4(),                        # synthetic, never persisted; see note below
    project_id=eval_run.project_id,
    organization_id=eval_run.organization_id,
    request_metadata=None,
    langfuse_credentials=None,
    include_provider_raw_response=True,    # file_search chunks for the knowledge_base metric
    include_guardrail_metadata=True,       # per-row guardrail record, gated on outcome.applied
    record_call=False,                     # no LlmCall row, no AI spans, no LLM metrics
)
```

- `job_id`: a fresh `uuid4()` per row. It is never persisted with `record_call=False`; its only
  live uses are log lines, the guardrails `request_id`, and the rephrase fallback
  `provider_response_id`. Per-row makes a guardrails-service log line traceable back to a single
  eval row for free. (Chains already reuse one `job_id` across blocks, so upstream tolerates
  either.)
- **Build a fresh `QueryParams` per row.** `execute_llm_call` mutates `query.input.content.value`
  in place, twice — once for `prompt_template` interpolation, once for the guardrails'
  `safe_text`. A reused object gets the template applied repeatedly.
- Resolve the config by stored reference (`id` + `version`), not as an ad-hoc blob, so eval goes
  through the exact `resolve_config_blob` path prod does.

### Result mapping

Add one field, `guardrail: str | None`, to the `ResponseResult` TypedDict and to
`fast_results.build_response_result`.

| `BlockResult` | `generated_output` | `failed` | `guardrail` |
|---|---|---|---|
| success | response text | `False` | `None`, or `"applied"` when `result.metadata` carries guardrail keys |
| `guardrail_outcome="blocked"` | `""` | **`False`** | `f"blocked: {result.error}"` |
| `guardrail_outcome="rephrased"` | the rephrase text | `False` | `"rephrased"` |
| any other `error` | `f"ERROR: {result.error}"` | `True` | `None` |

- `usage`: `extract_usage(result.usage, RESPONSE_USAGE_KEYS)` works unchanged —
  `RESPONSE_USAGE_KEYS` is exactly `("input_tokens", "output_tokens", "total_tokens")`, which is
  `models/llm/response.py::Usage`, and `field_value` reads attributes or dict keys.
- `response_id`: `result.response.response.provider_response_id`.
- `retrieved_chunks`: new `extract_file_search_chunks(raw: dict)` in
  `backend/app/crud/evaluations/response_parsing.py`, reading
  `result.response.provider_raw_response`. The existing
  `services/response/response.py::get_file_search_results` only works on the live SDK object;
  `include_provider_raw_response` hands back `response.model_dump()`, a plain dict. Keep emitting
  the same `{score, text, filename}` dicts so the S3 unit stays JSON-serializable.

### Retry

`_responses_call_for_item` is wrapped in `crud/evaluations/retry.py::retry_openai_call` (tenacity,
3 attempts, on `RateLimitError` / `APITimeoutError` / `APIConnectionError` /
`InternalServerError`). `execute_llm_call` **never raises** on provider failure — it returns
`BlockResult(error=...)` — so that retry is lost on the swap. On a 100-row burst against a
rate-limited project that is a real regression.

Retry `_llm_call_for_item` up to 3 times with exponential backoff whenever
`result.error is not None and result.retryable`. `error` is one string for every failure kind,
so the classification has to travel on its own field: `execute_llm_call` sets
`BlockResult.retryable` only where the provider, the proxy or the `LlmCall` write actually
failed. A guardrail block is never retried (content decision), and neither is a deterministic
failure — unresolvable config, rejected model, revoked key, guardrails auth fail-closed — since
three attempts and their backoff only reach the same answer more slowly. Known ceiling: a
provider 4xx is still flagged retryable, because the provider flattens its exception into a
string before `jobs.py` sees it; the waste is bounded (no tokens are billed on a rejected
request).

---

## Phase 3 — guardrail outcomes in scoring

**DONE.** Built as specced, with two deliberate deviations from the text below:

- **The unscoreable reason is a stable key, not the row's raw string.** The plan asked for the
  breakdown to read `{"blocked: ...": N}`. `merge.py::summarize_unscoreable` buckets any reason
  outside `UNSCOREABLE_REASONS` as `"other"`, so that would have produced exactly the
  undifferentiated bucket the plan wanted to avoid. New
  `score.py::UNSCOREABLE_GUARDRAIL_BLOCKED = "guardrail_blocked"`, added to that tuple. The provider
  message stays on the row's `guardrail` field and reaches the reviewer through the trace.
  (`test_blocked_rows_land_in_their_own_summary_bucket` is that assert; it produces `{"other": 2}`
  when the constant is patched out of the tuple.)
- **"Verify the v2 path records dropped rows" needed no v2 code, and `judge_stage.py` is untouched.**
  Both scoring paths already route through `fast_cosine.py::classify_empty_side` — v1 via
  `score_cosine_run`, v2 via `_score_judge_path` — so putting the guardrail check in as its **first**
  branch (ahead of `empty_output`) covers both. The v1 threshold fix is then one clause on
  `_stage2_embeddings`'s `embed_candidates`; `total_failures` and the `len(response_results)`
  denominator are untouched, since excluding the rows from the candidates already keeps them out of
  the numerator.

Other things worth knowing:

- `GUARDRAIL_BLOCKED/REPHRASED/APPLIED` moved from `crud/evaluations/fast.py` to `fast_results.py`
  so `fast_cosine.py` can share the new `is_guardrail_blocked()` predicate without an import cycle
  (`fast.py` already imports from `fast_cosine`). They lost their `GuardrailOutcomeLabel`
  annotation in the move — keeping it would have dragged an `app.services.llm` import into
  `fast_results.py`.
- The predicate matches the prefix `f"{GUARDRAIL_BLOCKED}:"` **with the colon**, and reads
  `.get("guardrail")` — S3 response units written before Phase 2 have no such key at all.
- `merge.py::_merge_single_trace` had to name `guardrail` explicitly: it rebuilds the merged dict
  from a fixed key list, so an unnamed key is dropped on **every** read, including a plain cache
  serve (which runs a step-forward merge with an empty fresh side). The carry is
  fresh-authoritative-when-present, not the `or` chain `category` uses — an absent/`None` guardrail
  means "did not fire", so an `or` would pin a stale `"blocked"` on a row that passes in a later
  run. A Langfuse-fetched trace (`langfuse.py:518`) omits the key entirely, so a resync preserves
  the cached value. Neither an `or` chain nor an unconditional `fresh` satisfies both directions;
  `TestMergeGuardrailOutcome` holds that pair.
- An all-blocked v1 run now reaches Stage 2 with `embed_candidates == []`. Verified safe:
  `_run_in_pool([])` returns `[]` and the empty unit uploads as `"[]"`, so the stage completes and
  writes its retry-skip marker normally.
- The v2 `setdefault` clause that would stop `judge_failed` overwriting `guardrail_blocked` is
  **unreachable for a blocked row** — `select_judgeable_rows` drops empty-output rows before
  judging, so a blocked row never enters `judge_failed_refs`. Tested as the real scenario instead:
  the two reasons coexist across different rows.
- Tests: 14 functions / 18 IDs added, 0 deleted, 0 rewritten. The primary regression red is real —
  removing the `is_guardrail_blocked` clause from `embed_candidates` raises
  `RuntimeError: Fast eval Stage 2 exceeded failure threshold | failed=3/4 | threshold=0.5`.
- **Known stale, deliberately not touched:** the `unscoreable` JSONB column comment at
  `models/evaluation.py:375` still enumerates the four old reasons. Alembic autogenerate diffs
  column comments, so fixing it needs a migration this change does not otherwise want.
- **Not in scope, worth a follow-up:** `crud/evaluations/embeddings.py:119` (the **batch** path)
  hardcodes `"empty_output"`/`"empty_ground_truth"` instead of calling `classify_empty_side`.
  Harmless today — batch rows never carry `guardrail` — but the two paths can now drift.


A blocked row lands with `generated_output=""`. On **v2** that is already enough:
`judge_stage.select_judgeable_rows` keeps only rows with a non-empty `generated_output`, and
Stage-1's failure threshold counts `failed` only, which a blocked row is not.

**v1 is not safe as-is — this is the one real gap.** `_stage2_embeddings` has its **own**
threshold check:

```python
embed_candidates = [r for r in response_results if not r.get("failed")]
...
total_failures = failed_count + sum(1 for r in response_results if r.get("failed"))
if is_failure_threshold_breached(failed_rows=total_failures, total_rows=len(response_results)):
    raise RuntimeError("Fast eval Stage 2 exceeded failure threshold | ...")
```

A blocked row has `failed=False`, so it enters `embed_candidates`, `_embedding_call_for_pair`
returns `build_embedding_failure` on the empty output, and it counts toward `total_failures`. A v1
run where guardrails block more than `EVAL_FAST_FAILURE_THRESHOLD` (0.5) of rows would **fail** —
contradicting the chosen behaviour.

Fix: exclude guardrail-blocked rows from `embed_candidates` and record them straight into
`unscoreable` with their guardrail reason, so they are neither embedded nor counted.

Also:

- **Carry the guardrail reason into the unscoreable entry.** `fast_cosine` writes the generic
  `"empty output or ground_truth"` reason today; prefer the row's `guardrail` value when set, so
  `merge.summarize_unscoreable` reports `{"blocked: ...": N}` instead of an undifferentiated
  "empty output" bucket.
- **Verify the v2 path records dropped rows.** Confirm rows filtered out by
  `select_judgeable_rows` end up in `eval_run.unscoreable` with a reason; add it in
  `_stage3_score_and_trace` if they currently vanish silently.
- **Surface the guardrail field on the trace** in `fast_traces.py`, so a reviewer can see per row
  whether guardrails fired.

### Known gap, deliberately not closed

A **bypassed** run (guardrails service unreachable → fail open, `guardrails.md` §4) is invisible to
the caller: metadata emission is gated on `outcome.applied`, and `apply_input_guardrails` /
`apply_output_guardrails` don't propagate the bypass flag. So an eval row can silently skip
guardrails and look identical to one that passed. Carrying a `"bypassed"` value through both
adapters is a follow-up; note it in the wiki rather than widening this change.

---

## Files

| File | Change |
|---|---|
| `backend/app/services/llm/chain/types.py` | `BlockResult.guardrail_outcome` |
| `backend/app/services/llm/jobs.py` | `record_call` flag (row + spans + metrics); gate the 3 creators; add the missing proxy `if llm_call_id:`; set `guardrail_outcome` at the 3 guardrail returns; carry `usage` on output blocks |
| `backend/app/services/llm/mappers.py` | `include=["file_search_call.results"]` in `transform_kaapi_config_to_native`'s OpenAI branch when a `file_search` tool is present **and** the new `include_file_search_results` flag is set; `execute_llm_call` drives it off `include_provider_raw_response` so production traffic is unchanged |
| `backend/app/crud/evaluations/response_parsing.py` | `extract_file_search_chunks(raw: dict)` |
| `backend/app/crud/evaluations/fast_results.py` | `guardrail` on `ResponseResult` + `build_response_result` |
| `backend/app/crud/evaluations/fast.py` | `_llm_call_for_item` replaces `_responses_call_for_item`; `run_response_chunk` takes `ConfigBlob`; `_stage2_embeddings` excludes blocked rows from `embed_candidates` and the threshold |
| `backend/app/services/evaluations/fast.py` | `_resolve_config_and_clients` returns the blob; `{{input}}` template guard in `validate_fast_evaluation_inputs` |
| `backend/app/crud/evaluations/fast_cosine.py`, `fast_traces.py`, `judge_stage.py` | guardrail reason into unscoreable + traces |

No migration — no schema change.

---

## Risks

1. **`prompt_template` starts being applied to eval rows.** Today eval sends the raw question and
   never interpolates; `execute_llm_call` does `template.replace("{{input}}", value)`. A config
   whose template omits `{{input}}` would send the template alone and **drop the question**,
   silently producing a run of answers to nothing. Hence the validation gate in Phase 2.
   (Fixing the interpolation gap is itself a win: `judge_stage` already shows the template to the
   judge as "the prompt wrapped around each user input", so the judge has been grading against a
   prompt the model was never given.) The iteration loop is safe: `prompt_improvement` only
   rewrites `params["instructions"]`, never `prompt_template`, so a config that passes the gate in
   round 1 keeps passing it.
2. **Latency.** Guardrails add up to two HTTP hops per direction (config fetch 10s timeout,
   validation 45s). Chunk budget today is 50 rows / 4 workers / 300s
   `CELERY_TASK_SOFT_TIME_LIMIT`. A guardrailed run can trip the soft limit; the cron healer then
   re-enqueues the chunk, and the `raw_output_url` skip guard means it re-charges the provider for
   the rows it hadn't written. Deferred by decision — watch `run_evaluation_fast_chunk` durations
   after rollout and tune `EVAL_FAST_CHUNK_SIZE` down first.
3. **DB reads per row.** `execute_llm_call` does a config-version read, a model-config validation
   read, `transform_kaapi_config_to_native`, and `get_llm_provider` (credentials) on every call —
   roughly four round trips per row, from four concurrent greenlets each holding a `Session`.
   Watch the connection pool. Caching the resolved provider per chunk is the obvious follow-up if
   it bites; don't pre-build it.
4. **`ChainBlock.execute` is a second caller** of `execute_llm_call` and must keep writing its
   `LlmCall` rows and its spans. `record_call` defaults to `True`, so it is untouched — the chain
   tests are the regression net.
5. **Fast mode is still gated to OpenAI + text** by `validate_fast_evaluation_inputs`.
   `execute_llm_call` supports every provider and input type, so that gate can be widened
   afterwards for free. Not in this change.

---

## Verification

1. `uv run bash scripts/tests-start.sh` — whole suite.
2. **Regression net for Phase 1:** `app/tests/services/llm/test_jobs.py`, `test_chain.py`,
   `test_chain_executor.py`, `test_guardrails.py` must pass **unchanged**. `record_call` defaults
   to `True`, so any diff there means the flag leaked into the default path.
3. New tests (`test-writer`):
   - `execute_llm_call(record_call=False)` creates no `LlmCall` row, returns `llm_call_id=None`,
     emits no `record_llm_call_started/finished`, and still returns the provider response —
     including down the rephrase and proxy branches.
   - An output-guardrail block still returns `usage`.
   - `guardrail_outcome` is `"blocked"` on an input hard block, `"blocked"` on an output hard block
     (assert **both** output call sites), `"rephrased"` on the rephrase path, `None` on a provider
     error.
   - `_llm_call_for_item` maps each `BlockResult` shape to the `ResponseResult` row in the table
     above; a blocked row has `failed=False` and an empty `generated_output`.
   - A chunk where every row is blocked completes without tripping `EVAL_FAST_FAILURE_THRESHOLD` —
     assert for **both** v1 (Stage-2 embeddings) and v2 (Stage-1 merge) — and the rows land in
     `eval_run.unscoreable` with the guardrail reason.
   - `transform_kaapi_config_to_native` emits `include` only when a `file_search` tool is present,
     and `map_kaapi_to_openai_params` output is unchanged for the batch/judge/assessment callers.
   - `validate_fast_evaluation_inputs` rejects a `prompt_template` without `{{input}}`.
   - Mock the guardrails HTTP boundary with `unittest.mock.patch` on
     `app.services.llm.guardrails.httpx.Client` — the module's established style, not `respx`
     (`guardrails.md` §13).
4. **End-to-end, against a dev project:** save a config with `input_guardrails` +
   `output_guardrails` set, `POST /api/v2/evaluations` with a small dataset, and check: the run
   reaches `completed`; **no new `llm_call` rows** — snapshot
   `select count(*) from llm_call where project_id = :p` before and after, since the synthetic
   `job_id` is never persisted and can't be queried; the S3 responses unit carries `guardrail` per
   row; blocked rows appear in `unscoreable` and not in the failure count; a knowledge-base config
   still produces `retrieved_chunks` so the `knowledge_base` judge metric scores.
5. Confirm no Langfuse generation is created for the run (v2 already passes `langfuse=None`), and
   that no `gen_ai.chat` span or LLM metric appears for the eval run in Sentry/OTel.

---

## Wiki updates (same PR — `CLAUDE.md` maintenance rule)

- `docs/wiki/modules/evaluations.md` — the "Eval traffic deliberately bypasses `/llm/call`" gotcha
  is now false for fast mode; record the guardrails behaviour and the unscoreable rule.
- `docs/wiki/modules/guardrails.md` — §1 and §6: eval fast runs are a third caller of the inline
  hooks. Add the bypass-invisibility gap to §12.
- `docs/wiki/modules/llm-call.md` — `record_call` and `guardrail_outcome` on the
  `execute_llm_call` contract; the new `include` behaviour in `transform_kaapi_config_to_native`.
- `docs/architecture/kaapi-evaluations-ARCHITECTURE.md` §12.1/§12.2 describe this as unbuilt —
  worth a note that fast mode now does it. (Those docs are already stale in places: §3.4 claims
  eval doesn't reuse the mappers; it does.)

---

## Verification run — 2026-09-27

### 1. Clean-DB suite — PASS

The local `ai_platform_test` on Homebrew pg14 (`localhost:5432`, *not* the docker pg17) was
stamped `083` with 083's column missing, which is what produced every `UndefinedColumn` failure
in earlier runs. It was left untouched — it may belong to another branch. A fresh database was
used instead:

```bash
createdb -h localhost -U postgres ai_platform_test_eg
cd backend && POSTGRES_DB=ai_platform_test_eg uv run bash scripts/tests-start.sh
```

Result: **3570 passed, 6 skipped, 0 failed** in 177s. All 8 previously drift-blocked IDs are
green, including `TestStage2GuardrailBlocked` (the Phase-3 regression test, which until now had
only ever passed under an in-transaction `ALTER TABLE` shim). The new DB reads `085`; the old one
still reads `083`.

Regression contract (Verification item 2) holds: `test_guardrails.py` is +76 lines with zero
deletions, so no existing assertion was bent. The one changed line in
`test_load_run_dataset_items.py` tracks `_resolve_config_and_clients`' new return arity, which is
a deliberate Phase-2 contract change.

### 2. `/pr-review` — approve with nits

Run against `git diff ccaebf7f` (working tree; nothing is committed, so `$BASE...HEAD` is empty
and the command's pull step was skipped — no upstream). 29 tracked files + 7 untracked,
1079+/410-. **No blocking issues.** Findings:

*Suggestions — fixed*

- `crud/evaluations/fast.py::run_response_chunk` read the model with an inline
  `isinstance(dict) / getattr` dance → now `field_value(config_blob.completion.params, "model")`,
  the helper already sitting one module over.
- `services/evaluations/fast.py` raised
  `detail=f"{ERR_CONFIG_TEMPLATE_MISSING_INPUT}: <prose>"` while every sibling in the same
  function raises the bare code → now `detail=ERR_CONFIG_TEMPLATE_MISSING_INPUT`, with the prose
  moved into a `logger.warning`, so a client matching `detail == "<code>"` matches.
- A guardrails auth fail-closed is no longer retried at all. It still fails the row, but the
  credentials are as broken on the third attempt as on the first, and on the *output* side each
  attempt would re-charge the provider, since the completion is generated before output
  guardrails run. This is what `BlockResult.retryable` exists for.

*Suggestion — deferred*

- `crud/evaluations/retry.py` imports `app.services.llm.chain.types`, deepening the existing
  crud → services dependency (`fast.py` already imported `services.llm.mappers` at HEAD).
  `[follow-up]`, not this PR.

*Nits — fixed*

- `GUARDRAIL_METADATA_KEYS` moved from `fast.py` to `fast_results.py`, beside the other
  `GUARDRAIL_*` constants.
- `_last_llm_result`'s bare `assert` replaced with an explicit `RuntimeError` guard.

*Checked and clean*

- `GuardrailsOutcome.blocked` (`error is not None and not raw.get("auth_error")`) was audited
  against every return in `run_guardrails_validation`: a 2xx body (the service's own verdict),
  the 401/403/422 branch (sets `auth_error`), and every other exception (returns
  `bypassed: True`, which `apply_guardrails` converts to `error=None`). There is no third
  fail-closed shape, so nothing that isn't a content verdict can be labelled `blocked`. A
  `list_validators_config` auth failure raises `ValueError` and surfaces as a plain row failure,
  which is the intended behaviour.
- `test_guardrails.py` is +76 lines with zero deletions; no existing assertion was bent.
- `TestRetryOpenAICall` (3 tests) dropped on the user's call: `retry_openai_call` is tenacity's
  stock exception-based behaviour and this branch does not change it. `test_retry.py` now covers
  `retry_llm_call` only; its `httpx` / `openai` / `retry_openai_call` imports went with it.
- Wiki maintenance rule satisfied: `evaluations.md`, `llm-call.md`, `domain-map.md`, `INDEX.md`,
  a new `guardrails.md`, the architecture doc and the `get_evaluation.md` API doc all move with
  the code.
- No migration, no schema change, no secrets in the diff.

### 3. End-to-end — two runs against a dev project

The docker stack was rebuilt from this working tree (`backend:latest`) and `backend` +
`celery-worker` recreated; both were verified to be running branch code before either run. No
Celery beat runs locally, so the cron barrier never enqueues the fan-in — `execute_fast_evaluation_aggregate`
was called directly inside the worker container for both runs. That code is untouched by this
branch, so it is a disclosure, not a gap.

**Run 18** — dataset `aicohort` (10 rows), config `cohort` **v3** (no guardrails):

- Reached `completed`; 10 rows generated, judged and scored; `unscoreable` empty.
- **No new `llm_call` rows.** `count(*) where project_id=1` was 1 before and 1 after, while a
  plain `POST /llm/call` on the same config moved it 0 → 1 as the positive control.
  `record_call=False` holds across 10 real provider calls.
- The S3 responses unit carries `guardrail` on **every** row (`None` here), with `failed=False`
  and real `usage`.
- `GET /evaluations/18?get_trace_info=true&export_format=row` returns `guardrail` on all 10
  traces, so the key survives `_merge_single_trace` and reaches the client.

**Run 19** — same dataset, config **v4** = v3 plus `input_guardrails` + `output_guardrails`
pointing at a synthetic validator id. Config-version save does **not** resolve validator ids
against the guardrails service, so a guardrailed blob is buildable even while that service is
down.

- The worker logged `[list_validators_config] Guardrails service unavailable ... Proceeding
  without input/output guardrails` and `[apply_guardrails] No validator configs resolved
  upstream` **per row, both directions** — live proof that fast eval now reaches the inline
  guardrail hooks.
- Fail-open behaved as `guardrails.md` §4 describes: run `completed`, 10/10 rows answered,
  `failed=False`, `unscoreable` empty, `llm_call` still 1.
- Every row's `guardrail` is `None`, **indistinguishable from a run where guardrails passed.**
  That is the bypass-invisibility gap the plan records under "Known gap, deliberately not
  closed", now observed rather than reasoned about.

**Still not covered:**

- A real *block*. The local `kaapi-guardrails-backend` container never finishes booting — its
  Guardrails Hub token has expired, so `hub://guardrails/ban_list` fails to install and nothing
  listens on its port (`curl` returns `000` from the host and from inside the worker network).
  Proving `guardrail="blocked: ..."` and a `guardrail_blocked` unscoreable entry needs a fresh
  hub token in that repo, not a change here.
- The v1 Stage-2 embedding threshold fix. `/api/v2/evaluations` always takes the judge path.
  `TestStage2GuardrailBlocked` is its only proof, and it passes.
- Verification item 5 (no `gen_ai.chat` span or LLM metric in Sentry/OTel for the run).

**Leftovers, not cleaned up:** the `ai_platform_test_eg` database, config `cohort` v4, runs 18
and 19, and the positive-control `llm_call` row plus its job.
