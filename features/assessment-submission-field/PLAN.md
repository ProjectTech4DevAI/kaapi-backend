# Assessment `submission` Field + Input-Schema Validation — Implementation Plan

Source spec: verbal description (meeting note), see Open Questions.

## Summary

Move the per-row prompt template off the API request and into the ASSESSMENT
`config_blob`. Each `config_blob` block gets its own `submission` string (the
`{column}` template previously carried by the request's `query`): mandatory on
`assessment`, optional on each pre-filter. At config save the blob validator
checks every `{placeholder}` in every `submission` resolves against the keys
declared in `assessment.params.input_schema`, rejecting the save otherwise. The
request models drop `query` entirely; the BATCH pipeline sources its per-row
prompt from the config instead.

## Blast Radius

Primary entities: Config / ConfigVersion (ASSESSMENT `config_blob`), Assessment (BATCH request input).

| Surface | Hop | Impact | Decision |
|---|---|---|---|
| Config / ConfigVersion (`config_blob`) | 0 | New `submission` field per block; new save-time placeholder validation | in scope |
| Config save path (`crud/config/config.py`, `version.py`, `crud/model_config.py::validate_config_blob_for_tag`) | 1 | Validation rides existing `AssessmentConfigBlob.model_validate`; no call-site change | in scope (no code change) |
| `POST /assessments` request (`BatchInput`, `ResponseInput`) | 1 | `query` removed from both; breaking for API clients | in scope |
| BATCH pipeline (`services/assessment/api/batch.py`) | 1 | Prompt now read from config `submission`, not request `query` | in scope |
| Provider Batch APIs (OpenAI/Gemini/Anthropic) | 2 | Prompt string still a plain per-row text; payload shape unchanged | unaffected |
| kaapi-frontend console (config CRUD + submit form) | 2 | Config gains `submission`; request loses `query` — frontend must follow | out of scope (separate repo; flagged in PR) |
| Langfuse | 2 | No trace-shape change | unaffected |
| Object storage | 2 | Attachment handling unchanged | unaffected |
| Legacy RUN (`InputBinding.prompt`) | 2 | Dataset-driven, separate template; untouched | out of scope |

## Steps

### 1. Model: add `submission` to config params + blob-level placeholder validator
- Files: `app/models/config/assessment_blob.py` (change)
- Add `submission: str | None` (optional) to `PreFilterParams`.
- Add `submission: str` (mandatory, `min_length=1`) to `AssessmentTextParams`.
- Add a `PLACEHOLDER_RE` module constant (`re.compile(r"\{(\w+)\}")`) and a
  `model_validator(mode="after")` on `AssessmentConfigBlob` that:
  - reads declared keys from `self.assessment.params["input_schema"]`,
  - collects placeholders from `self.assessment.params["submission"]` and, when
    present, each pre-filter's `params.get("submission")`,
  - raises `ValueError` naming the block + the unknown placeholder(s) if any
    placeholder is not a declared input-schema key.
  Validator lives at blob level because pre-filter submissions validate against
  the *assessment* block's `input_schema` (cross-block visibility).
- Depends on: nothing

### 2. Request models: drop `query`
- Files: `app/models/assessment/assessment_api.py` (change)
- Remove the `query` field from `BatchInput` (keeps `data`) and from
  `ResponseInput` (keeps `attachments`); update both docstrings. `extra=forbid`
  now rejects a stray `query`, so old requests fail loudly.
- Depends on: nothing

### 3. Service: source the prompt from config, keep `submission` out of provider params
- Files: `app/services/assessment/api/batch.py` (change)
- `_stage_prompt(batch_input, stage)` → `_stage_prompt(blob, stage)`:
  - `ASSESSMENT` stage returns `blob.assessment.params["submission"]`,
  - a pre-filter stage returns `_prefilter_for_stage(blob, stage).params.get("submission")`
    (None when the pre-filter declares no template),
  - update the call site (currently `prompt=_stage_prompt(batch_input, stage)`).
- `_stage_params`: pop `submission` from the returned params in BOTH the
  assessment and pre-filter branches (same treatment as `input_schema` — a
  request/prompt concern, not a provider param).
- Depends on: steps 1, 2

### 4. Test fixtures / example configs
- Files: `z_assessment_test/tap_test/config_openai.json`,
  `z_assessment_test/tap_test/input.json` (change)
- config already carries `submission` in both blocks; drop the stale
  `// new schema validate...` comment. input.json already has `query` removed
  (commented out) — delete the dead lines.
- Depends on: steps 1, 2

### 5. Wiki
- Files: `docs/wiki/modules/assessment.md` (change)
- Tables paragraph: config blob now owns `submission` per block (mandatory on
  assessment, optional per pre-filter), validated against `input_schema` at
  save; request `BatchInput` is `{data}` and `ResponseInput` is `{attachments}`
  — neither carries `query` any longer. Pre-filters now can carry a per-row
  `submission` template (previously none).
- Depends on: steps 1-3

## Migration

None. `config_blob` is JSONB — no DDL, no column change. Existing ASSESSMENT
configs are not re-validated on read, so stored rows are unaffected until their
next save (which will then require `submission`). Note this in the PR body.

## Tests

`app/tests/assessment/` (test-writer, after phase 1):
- Config save: valid `submission` (assessment + pre-filter) passes; unknown
  placeholder in assessment `submission` rejected; unknown placeholder in a
  pre-filter `submission` rejected; assessment `submission` missing rejected;
  pre-filter `submission` absent is accepted.
- Request: `BatchInput`/`ResponseInput` with a `query` key now rejected
  (`extra=forbid`); without it accepted.
- BATCH pipeline: `_stage_prompt` returns the config `submission` for the
  assessment stage and the pre-filter's `submission` (or None) for a pre-filter
  stage; `_stage_params` output contains no `submission` key.
- Update existing fixtures in `test_api_batch.py`, `test_api_submission.py`,
  `test_topic_relevance.py`, `test_api_crud.py`, `test_service.py` to drop
  request `query` and add `submission` to blob fixtures.
- Mock the provider batch HTTP boundary as the existing batch tests already do;
  no live provider calls.

## Open Questions

- Spec is a verbal meeting note. Assumptions confirmed with the user:
  `submission` optional on pre-filters / mandatory on assessment; `query`
  removed from both `BatchInput` and `ResponseInput`; frontend handled in a
  separate follow-up (kaapi-frontend repo), this PR is backend-only.
- RESPONSE has no `input_schema`, so its config-side `submission` story is
  undefined; RESPONSE is a route-level 501 stub, so this is deferred until
  RESPONSE is wired.
