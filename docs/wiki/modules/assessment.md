# Module: Assessment

Batch assessments of datasets against a config (prompt_template + text columns + attachments over rows). No deep-dive doc yet.

All paths relative to `backend/app/`.

## Routes
- `api/routes/assessment/`

## Tables (SQLModel)
| Table | Model |
|---|---|
| `assessment` (Assessment; FK → config, evaluation_dataset, batch_job, self-parent) | `models/assessment.py` |
| `assessment_run` (AssessmentRun) | `models/assessment.py` |

## Services / CRUD
- `services/assessment/`
- `crud/assessment/` — `batch.py` builds per-row prompts (`prompt_template` placeholder substitution `{column_name}`)

## Async
- Rides shared `core/batch/` provider batch infra + cron polling (same as evaluations).

## External
- Provider Batch APIs, object storage for attachments.
