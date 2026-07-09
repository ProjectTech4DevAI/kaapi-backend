"""AI-assisted prompt improvement service.

Split into a fast request-side path (validate + enqueue a Celery job) and a
worker-side path that does the heavy Anthropic round-trip. The LLM call stays
synchronous — blocking in a Celery worker is fine, and it holds no FastAPI
threadpool thread.

Loads the evaluation run's stored score traces from object storage, asks Claude
to rewrite the system prompt, and persists the result as a new config_version.
"""

import copy
import json
import logging
from uuid import UUID

import anthropic
from asgi_correlation_id import correlation_id
from celery.exceptions import SoftTimeLimitExceeded
from fastapi import HTTPException
from gevent import Timeout
from sqlmodel import Session

from app.celery.utils import start_prompt_improvement
from app.core.cloud.storage import get_cloud_storage
from app.core.config import settings
from app.core.db import engine
from app.core.storage_utils import load_json_from_object_store
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.core import get_evaluation_run_by_id
from app.crud.jobs import JobCrud
from app.models.config.config import ConfigTag
from app.models.config.version import ConfigVersion, ConfigVersionUpdate
from app.models.evaluation import EvaluationRun
from app.models.job import Job, JobStatus, JobType, JobUpdate
from app.services.llm.providers.claude import ClaudeProvider

logger = logging.getLogger(__name__)

# Headroom for a full prompt rewrite + JSON wrapper; too low truncates into invalid JSON.
_LLM_MAX_TOKENS = 16384

# JSON keys expected in the LLM's structured response.
_LLM_KEY_INSTRUCTIONS = "improved_instructions"
_LLM_KEY_RATIONALE = "rationale"

# Cap the rationale so it stays a one-line summary in the commit_message.
_RATIONALE_MAX_LENGTH = 200

# Structured-output schema: guarantees the response is valid JSON with these fields.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        _LLM_KEY_INSTRUCTIONS: {"type": "string"},
        _LLM_KEY_RATIONALE: {"type": "string", "maxLength": _RATIONALE_MAX_LENGTH},
    },
    "required": [_LLM_KEY_INSTRUCTIONS, _LLM_KEY_RATIONALE],
    "additionalProperties": False,
}

COMMIT_MESSAGE_MAX_LENGTH = 512

# Prefix that marks a commit_message as AI-generated; used as a search token for
# audit queries and by the test suite to assert provenance.
AI_GENERATED_MARKER = "[AI Generated]"

_COMPLETED_STATUS = "completed"


def _resolve_source_version(
    *,
    session: Session,
    run: EvaluationRun,
    project_id: int,
) -> ConfigVersion | None:
    """Resolve the exact config_version the run evaluated.

    read_one() raises 404 when the config itself is missing/soft-deleted and
    returns None when only the version is gone — the caller treats both as the
    run's source config no longer resolving.
    """
    try:
        return ConfigVersionCrud(
            session=session,
            config_id=run.config_id,
            project_id=project_id,
            tag=ConfigTag.DEFAULT,
        ).read_one(version_number=run.config_version)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        return None


def validate_improve_prompt(
    *,
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationRun:
    """Run the cheap DB precondition checks for prompt improvement.

    Raises HTTPException for every domain failure so the request-side caller
    returns a real 4xx before any job is enqueued. No LLM call, no trace
    download. Returns the run for reuse by the worker path.
    """
    run = get_evaluation_run_by_id(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        project_id=project_id,
    )
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="evaluation_not_found: no evaluation run with this id in the caller's project",
        )
    if run.status != _COMPLETED_STATUS:
        raise HTTPException(
            status_code=409,
            detail=f"evaluation_not_completed: run status is '{run.status}', must be '{_COMPLETED_STATUS}'",
        )
    if not run.score_trace_url:
        # Run exists but a precondition is unmet, so 422 rather than 404.
        raise HTTPException(
            status_code=422,
            detail=(
                "traces_not_available: this run has no score_trace_url; "
                "cannot improve prompt"
            ),
        )
    if run.config_id is None or run.config_version is None:
        # Both FKs are nullable; without them there's no version to improve.
        raise HTTPException(
            status_code=422,
            detail="source_config_unavailable: run has no config_id/config_version reference",
        )

    if _resolve_source_version(session=session, run=run, project_id=project_id) is None:
        raise HTTPException(
            status_code=409,
            detail="source_config_unavailable: the run's config or config_version is missing or soft-deleted",
        )

    return run


def start_prompt_improvement_job(
    *,
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
) -> Job:
    """Validate preconditions, create a job row, and enqueue the worker task.

    Returns the created Job immediately so the route can hand back a poll handle.
    """
    validate_improve_prompt(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    trace_id = correlation_id.get() or "N/A"
    job = JobCrud(session=session).create(
        job_type=JobType.PROMPT_IMPROVEMENT,
        trace_id=trace_id,
        project_id=project_id,
    )

    logger.info(
        f"[start_prompt_improvement_job] Job created | job_id={job.id} "
        f"evaluation_id={evaluation_id} project_id={project_id}"
    )

    try:
        task_id = start_prompt_improvement(
            project_id=project_id,
            job_id=str(job.id),
            trace_id=trace_id,
            organization_id=organization_id,
            evaluation_id=evaluation_id,
        )
    except Exception as exc:
        logger.error(
            f"[start_prompt_improvement_job] Failed to enqueue | job_id={job.id} "
            f"evaluation_id={evaluation_id} | {exc}",
            exc_info=True,
        )
        JobCrud(session=session).update(
            job.id,
            JobUpdate(
                status=JobStatus.FAILED,
                error_message=f"Failed to queue prompt improvement: {exc}",
            ),
        )
        raise HTTPException(
            status_code=500,
            detail="prompt_improvement_enqueue_failed: could not queue the job",
        )

    logger.info(
        f"[start_prompt_improvement_job] Enqueued | job_id={job.id} task_id={task_id}"
    )
    return job


def execute_prompt_improvement(
    *,
    project_id: int,
    job_id: str,
    organization_id: int,
    evaluation_id: int,
    **kwargs,
) -> dict:
    """Worker entrypoint: run the full prompt-improvement flow and record it on the Job.

    Opens its own session (the request session is long gone). On success the new
    config_version's id/version and the rationale land on Job.meta; on failure the
    Job is marked FAILED with a clean message and the error is re-raised so Celery
    records the failure.
    """
    task_id = kwargs.get("task_id")
    job_uuid = UUID(job_id)

    logger.info(
        f"[execute_prompt_improvement] Starting | job_id={job_id} "
        f"evaluation_id={evaluation_id} project_id={project_id}"
    )

    with Session(engine) as session:
        job_crud = JobCrud(session=session)

        # Idempotency (Celery redelivers): a SUCCESS job already minted a
        # config_version and paid for the LLM call, so re-running would double-spend
        # and create a duplicate version. A stuck PROCESSING job (worker died
        # mid-run) is intentionally allowed to re-run.
        existing = job_crud.get(job_id=job_uuid, project_id=project_id)
        if existing and existing.status == JobStatus.SUCCESS:
            logger.info(
                f"[execute_prompt_improvement] Redelivery of completed job, skipping | "
                f"job_id={job_id}"
            )
            return {"success": True, **(existing.meta or {})}

        job_crud.update(
            job_uuid, JobUpdate(status=JobStatus.PROCESSING, task_id=task_id)
        )

        try:
            # Re-validate defensively: the run may have changed between enqueue and pickup.
            run = validate_improve_prompt(
                session=session,
                evaluation_id=evaluation_id,
                organization_id=organization_id,
                project_id=project_id,
            )
            version = _resolve_source_version(
                session=session, run=run, project_id=project_id
            )
            if version is None:
                raise RuntimeError(
                    "source_config_unavailable: the run's config or config_version is missing or soft-deleted"
                )

            blob = version.config_blob or {}
            params = blob.get("completion", {}).get("params", {}) or {}
            current_instructions = params.get("instructions") or ""

            storage = get_cloud_storage(session=session, project_id=project_id)
            traces = load_json_from_object_store(
                storage=storage, url=run.score_trace_url
            )
            if traces is None:
                raise RuntimeError(
                    "trace_download_failed: could not retrieve trace file from storage"
                )

            improved_instructions, rationale = _draft_improved_prompt(
                current_instructions=current_instructions,
                config_params=params,
                traces=traces,
            )

            # Derive the new version from the *evaluated* version's blob (not the
            # latest active one) so model, knowledge base, and other params stay
            # apples-to-apples with what was scored — only the prompt text changes.
            improved_blob = copy.deepcopy(blob)
            improved_blob.setdefault("completion", {}).setdefault("params", {})[
                "instructions"
            ] = improved_instructions

            commit_message = (
                f"{AI_GENERATED_MARKER} improved from config version v{run.config_version} "
                f"(Evaluation: {run.run_name}) {rationale}"
            )[:COMMIT_MESSAGE_MAX_LENGTH]

            new_version = ConfigVersionCrud(
                session=session,
                config_id=run.config_id,
                project_id=project_id,
            ).create_or_raise(
                ConfigVersionUpdate(
                    config_blob=improved_blob,
                    commit_message=commit_message,
                )
            )

            job_crud.update(
                job_uuid,
                JobUpdate(
                    status=JobStatus.SUCCESS,
                    meta={
                        "config_version_id": str(new_version.id),
                        "version": new_version.version,
                        "rationale": rationale,
                    },
                ),
            )

            logger.info(
                f"[execute_prompt_improvement] Done | job_id={job_id} "
                f"evaluation_id={evaluation_id} new_version_id={new_version.id} "
                f"version={new_version.version}"
            )
            return {
                "success": True,
                "config_version_id": str(new_version.id),
                "version": new_version.version,
            }

        except (Timeout, SoftTimeLimitExceeded):
            logger.warning(
                f"[execute_prompt_improvement] Timeout | job_id={job_id} "
                f"evaluation_id={evaluation_id}"
            )
            job_crud.update(
                job_uuid,
                JobUpdate(
                    status=JobStatus.FAILED,
                    error_message="Task exceeded soft time limit",
                ),
            )
            raise
        except Exception as exc:
            # HTTPException (from re-validation) carries the clean detail; everything
            # else falls back to str(exc).
            error_message = getattr(exc, "detail", None) or str(exc)
            logger.error(
                f"[execute_prompt_improvement] Failed | job_id={job_id} "
                f"evaluation_id={evaluation_id} | {error_message}",
                exc_info=True,
            )
            job_crud.update(
                job_uuid,
                JobUpdate(status=JobStatus.FAILED, error_message=error_message),
            )
            raise


def _draft_improved_prompt(
    *,
    current_instructions: str,
    config_params: dict,
    traces: list | dict,
) -> tuple[str, str]:
    """Call Claude with the current prompt + score traces and return
    (improved_instructions, rationale).

    Uses structured outputs so the first text block is guaranteed-valid JSON.
    Runs inside a Celery worker, so failures raise plain exceptions (no client
    sees an HTTP status here) — the worker turns them into Job FAILED. The
    per-cause log lines are kept so operators can still triage the fault.
    """
    if not settings.ANTHROPIC_API_KEY:
        # Missing platform key is a server-side misconfiguration, not an upstream fault.
        raise RuntimeError(
            "prompt_generation_failed: the platform Anthropic key "
            "(ANTHROPIC_API_KEY) is not configured"
        )

    client = ClaudeProvider.create_client({"api_key": settings.ANTHROPIC_API_KEY})

    # instructions is shown above already; knowledge_base_ids are opaque ids the model can't use.
    excluded_keys = {"instructions", "knowledge_base_ids"}
    target_config = {
        key: value for key, value in config_params.items() if key not in excluded_keys
    }

    user_message_text = (
        "You are a prompt engineer. Below is a JSON array of evaluation traces. "
        "Each trace has the fields: `question`, `ground_truth_answer`, "
        "`llm_answer`, `category`, and `scores` (a list of scoring objects with "
        "`name`, `value`, and `unscoreable`).\n\n"
        f"## Evaluation traces\n```\n{json.dumps(traces)}\n```\n\n"
        f"## Current system prompt\n```\n{current_instructions}\n```\n\n"
        "## Target configuration (read-only — do NOT change any of these)\n"
        f"```\n{json.dumps(target_config)}\n```\n\n"
        "## Task\n"
        "1. Identify the answers that performed poorly — those with low scores or "
        "where `llm_answer` diverges significantly from `ground_truth_answer`.\n"
        "2. Rewrite the system prompt to improve those low-performing answers while "
        "keeping what already works well.\n"
        "3. Change ONLY the prompt text — do not alter the model, knowledge base, "
        "or any other configuration.\n"
        f"4. Return `{_LLM_KEY_RATIONALE}` as ONE concise sentence "
        f"(≤ {_RATIONALE_MAX_LENGTH} characters): what you changed and why."
    )

    try:
        response = client.messages.create(
            model=settings.PROMPT_IMPROVEMENT_MODEL,
            max_tokens=_LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": user_message_text}],
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return data[_LLM_KEY_INSTRUCTIONS], data[_LLM_KEY_RATIONALE]

    except anthropic.AuthenticationError:
        logger.warning(
            "[_draft_improved_prompt] [ANTHROPIC] Authentication failed "
            "(code: 401): Verify the ANTHROPIC_API_KEY is "
            "valid, not expired, and configured correctly.",
            exc_info=True,
        )
        raise RuntimeError(
            "prompt_generation_failed: Anthropic authentication failed — "
            "verify the platform API key is valid and not expired"
        )

    except anthropic.RateLimitError:
        logger.warning(
            "[_draft_improved_prompt] [ANTHROPIC] Rate limit exceeded "
            "(code: 429): Hit Anthropic rate/quota — wait ≥1 min and retry.",
            exc_info=True,
        )
        raise RuntimeError(
            "prompt_generation_failed: Anthropic rate limit exceeded — "
            "wait at least 1 minute and retry"
        )

    except anthropic.APITimeoutError:
        # Must come before APIConnectionError — APITimeoutError is a subclass.
        logger.error(
            "[_draft_improved_prompt] [KAAPI] Anthropic request timed out "
            "(code: APITimeoutError): retry with a smaller payload.",
            exc_info=True,
        )
        raise RuntimeError(
            "prompt_generation_failed: Anthropic request timed out — "
            "retry. If persistent, contact Kaapi"
        )

    except anthropic.APIConnectionError:
        logger.error(
            "[_draft_improved_prompt] [KAAPI] Anthropic connection failed "
            "(code: APIConnectionError): network or DNS issue reaching Anthropic.",
            exc_info=True,
        )
        raise RuntimeError(
            "prompt_generation_failed: network error reaching Anthropic — "
            "check connectivity. If persistent, contact Kaapi"
        )

    except anthropic.APIStatusError as exc:
        status = exc.status_code
        # 5xx is provider-side (alert-worthy); 4xx is caller's fault (noise if alerted)
        log = logger.error if status and status >= 500 else logger.warning
        log(
            f"[_draft_improved_prompt] [ANTHROPIC] API status error "
            f"(code: {status}): {exc.message}.",
            exc_info=True,
        )
        raise RuntimeError(
            f"prompt_generation_failed: Anthropic returned HTTP {status} — "
            "retry or contact Kaapi if persistent"
        )

    except Exception as exc:
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Unexpected error during LLM call "
            f"(code: {type(exc).__name__}): not raised by the Anthropic SDK — "
            f"likely a Kaapi-side failure. Contact Kaapi if persistent.",
            exc_info=True,
        )
        raise RuntimeError(
            "prompt_generation_failed: unexpected error during prompt generation — "
            "contact Kaapi if persistent"
        )
