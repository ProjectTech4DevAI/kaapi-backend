"""AI-assisted prompt improvement service.

Downloads the evaluation run's stored trace file from S3, hands it to Claude
via the Files API, and persists the rewritten system prompt as a new
config_version.
"""

import json
import logging
from typing import Any

import anthropic
from anthropic import Anthropic
from fastapi import HTTPException
from sqlmodel import Session

from app.core.cloud.storage import get_cloud_storage
from app.core.config import settings
from app.crud.config.config import ConfigCrud
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.core import get_evaluation_run_by_id
from app.models.config.config import ConfigTag
from app.models.config.version import (
    ConfigVersion,
    ConfigVersionPublic,
    ConfigVersionUpdate,
)
from app.models.evaluation import EvaluationRun

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

# Room for a full prompt rewrite plus structured JSON wrapper.
_LLM_MAX_TOKENS = 8192

# JSON keys expected in the LLM's structured response.
_LLM_KEY_INSTRUCTIONS = "improved_instructions"
_LLM_KEY_RATIONALE = "rationale"

COMMIT_MESSAGE_MAX_LENGTH = 512

# Prefix that marks a commit_message as AI-generated; used as a search token for
# audit queries and by the test suite to assert provenance.
AI_GENERATED_MARKER = "[AI Generated]"

# Anthropic Files API beta header value — required by client.beta.files.*
# and client.beta.messages.create(..., betas=[...]).
_FILES_API_BETA = "files-api-2025-04-14"

# Content-type for the uploaded trace file. text/plain is reliably accepted as
# a document block; application/json is sometimes rejected by the Files API.
_TRACE_CONTENT_TYPE = "text/plain"


# ── public entry point ────────────────────────────────────────────────────────


def improve_prompt(
    *,
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
) -> ConfigVersionPublic:
    """Run the full prompt-improvement flow synchronously and return the new version.

    Raises HTTPException for all domain errors so the route stays thin.
    """
    logger.info(
        f"[improve_prompt] Starting | evaluation_id={evaluation_id} "
        f"project_id={project_id}"
    )

    run = _load_completed_run(
        session=session,
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    if not run.score_trace_url:
        raise HTTPException(
            status_code=422,
            detail=(
                "traces_not_available: this run has no score_trace_url; "
                "cannot improve prompt"
            ),
        )

    source_version = _resolve_source_version(
        session=session,
        run=run,
        project_id=project_id,
    )

    current_instructions = _extract_instructions(source_version)

    # Pull the model name from the source config blob for LLM context.
    # It may be absent (e.g. older blobs), so treat it as optional.
    blob: dict[str, Any] = source_version.config_blob or {}
    model_name: str | None = (
        blob.get("completion", {}).get("params", {}).get("model") or None
    )

    trace_bytes = _download_trace_file(
        session=session,
        project_id=project_id,
        url=run.score_trace_url,
    )

    improved_instructions, rationale = _draft_improved_prompt(
        evaluation_id=evaluation_id,
        current_instructions=current_instructions,
        model_name=model_name,
        trace_bytes=trace_bytes,
    )

    commit_message = (
        f"{AI_GENERATED_MARKER} (source_evaluation_run_id={evaluation_id}) {rationale}"
    )[:COMMIT_MESSAGE_MAX_LENGTH]

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=run.config_id,
        project_id=project_id,
    )
    new_version = version_crud.create_or_raise(
        ConfigVersionUpdate(
            config_blob={
                "completion": {"params": {"instructions": improved_instructions}}
            },
            commit_message=commit_message,
        )
    )

    logger.info(
        f"[improve_prompt] Done | evaluation_id={evaluation_id} "
        f"new_version_id={new_version.id} version={new_version.version}"
    )

    return ConfigVersionPublic.model_validate(new_version)


# ── helpers: validation & data loading ───────────────────────────────────────


def _load_completed_run(
    *,
    session: Session,
    evaluation_id: int,
    organization_id: int,
    project_id: int,
) -> EvaluationRun:
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

    if run.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"evaluation_not_completed: run status is '{run.status}', must be 'completed'",
        )

    return run


def _resolve_source_version(
    *,
    session: Session,
    run: EvaluationRun,
    project_id: int,
) -> ConfigVersion:
    """Load the config + config_version referenced by the run; guard soft-deletes and tenant scope."""
    config = ConfigCrud(session, project_id).read_one(run.config_id)

    if config is None:
        raise HTTPException(
            status_code=409,
            detail="source_config_unavailable: the run's config is missing, soft-deleted, or outside the caller's project",
        )

    version_crud = ConfigVersionCrud(
        session=session,
        config_id=run.config_id,
        project_id=project_id,
        tag=ConfigTag.DEFAULT,
    )
    version = version_crud.read_one(version_number=run.config_version)

    if version is None:
        raise HTTPException(
            status_code=409,
            detail="source_config_unavailable: the run's config_version is missing or soft-deleted",
        )

    return version


def _download_trace_file(
    *,
    session: Session,
    project_id: int,
    url: str,
) -> bytes:
    """Fetch the trace JSON bytes from S3. Maps storage errors to 502."""
    try:
        storage = get_cloud_storage(session=session, project_id=project_id)
        trace_bytes = storage.get(url)
        logger.info(
            f"[_download_trace_file] Downloaded trace file | "
            f"project_id={project_id} url={url} size_bytes={len(trace_bytes)}"
        )
        return trace_bytes
    except Exception as exc:
        logger.error(
            f"[_download_trace_file] [KAAPI] Failed to download trace file "
            f"(code: {type(exc).__name__}): could not retrieve trace data from S3. "
            f"Check storage credentials and that the file exists. | url={url}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail="trace_download_failed: could not retrieve trace file from storage",
        )


# ── helpers: LLM call ─────────────────────────────────────────────────────────


def _build_improvement_prompt(
    *,
    current_instructions: str,
    model_name: str | None,
) -> str:
    """Build the text block sent alongside the trace file document."""
    model_context = (
        f"\n\nThe system prompt runs on model: `{model_name}`." if model_name else ""
    )

    return (
        "You are a prompt engineer. The attached file is a JSON array of evaluation "
        "traces. Each trace has the fields: `question`, `ground_truth_answer`, "
        "`llm_answer`, `category`, and `scores` (a list of scoring objects with "
        "`name`, `value`, and `unscoreable`).\n\n"
        f"## Current system prompt\n```\n{current_instructions}\n```"
        f"{model_context}\n\n"
        "## Task\n"
        "1. Identify the answers that performed poorly — those with low scores or "
        "where `llm_answer` diverges significantly from `ground_truth_answer`.\n"
        "2. Rewrite the system prompt to improve those low-performing answers while "
        "keeping what already works well.\n"
        "3. Change ONLY the prompt text — do not alter the model, knowledge base, "
        "or any other configuration.\n\n"
        "## Response format\n"
        "Respond ONLY with a JSON object (no markdown fences) with exactly two keys:\n"
        '  - "improved_instructions": the full rewritten prompt string\n'
        '  - "rationale": one short paragraph explaining what you targeted and why\n'
    )


def _draft_improved_prompt(
    *,
    evaluation_id: int,
    current_instructions: str,
    model_name: str | None,
    trace_bytes: bytes,
) -> tuple[str, str]:
    """Upload the trace file to the Anthropic Files API, call Claude, and return
    (improved_instructions, rationale).

    The uploaded file is always deleted after the call, even on error.
    Raises HTTPException(502) on any LLM failure or unusable output.
    """
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: the platform Anthropic key "
                "(ANTHROPIC_API_KEY) is not configured"
            ),
        )

    user_message_text = _build_improvement_prompt(
        current_instructions=current_instructions,
        model_name=model_name,
    )

    client = Anthropic(api_key=api_key)

    uploaded = client.beta.files.upload(
        file=(f"traces_{evaluation_id}.txt", trace_bytes, _TRACE_CONTENT_TYPE),
    )
    logger.info(
        f"[_draft_improved_prompt] Uploaded trace file to Anthropic Files API | "
        f"file_id={uploaded.id} evaluation_id={evaluation_id}"
    )

    try:
        response = client.beta.messages.create(
            model=settings.PROMPT_IMPROVEMENT_MODEL,
            max_tokens=_LLM_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message_text},
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": uploaded.id},
                        },
                    ],
                }
            ],
            betas=[_FILES_API_BETA],
        )
        raw_text = next((b.text for b in response.content if b.type == "text"), "")
        logger.info(
            f"[_draft_improved_prompt] LLM call succeeded | "
            f"model={settings.PROMPT_IMPROVEMENT_MODEL} response_id={response.id}"
        )

    except anthropic.AuthenticationError:
        logger.warning(
            f"[_draft_improved_prompt] [ANTHROPIC] Authentication failed "
            f"(code: 401): Verify the ANTHROPIC_API_KEY is "
            f"valid, not expired, and configured correctly.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic authentication failed — "
                "verify the platform API key is valid and not expired"
            ),
        )

    except anthropic.RateLimitError:
        logger.warning(
            f"[_draft_improved_prompt] [ANTHROPIC] Rate limit exceeded "
            f"(code: 429): Hit Anthropic rate/quota — wait ≥1 min and retry.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic rate limit exceeded — "
                "wait at least 1 minute and retry"
            ),
        )

    except anthropic.APITimeoutError:
        # Must come before APIConnectionError — APITimeoutError is a subclass.
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Anthropic request timed out "
            f"(code: APITimeoutError): retry with a smaller payload.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic request timed out — "
                "retry. If persistent, contact Kaapi"
            ),
        )

    except anthropic.APIConnectionError:
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Anthropic connection failed "
            f"(code: APIConnectionError): network or DNS issue reaching Anthropic.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: network error reaching Anthropic — "
                "check connectivity. If persistent, contact Kaapi"
            ),
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
        raise HTTPException(
            status_code=502,
            detail=(
                f"prompt_generation_failed: Anthropic returned HTTP {status} — "
                "retry or contact Kaapi if persistent"
            ),
        )

    except Exception as exc:
        logger.error(
            f"[_draft_improved_prompt] [KAAPI] Unexpected error during LLM call "
            f"(code: {type(exc).__name__}): not raised by the Anthropic SDK — "
            f"likely a Kaapi-side failure. Contact Kaapi if persistent.",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: unexpected error during prompt generation — "
                "contact Kaapi if persistent"
            ),
        )

    finally:
        try:
            client.beta.files.delete(uploaded.id)
        except Exception:
            logger.warning(
                f"[_draft_improved_prompt] failed to delete uploaded trace file | "
                f"file_id={uploaded.id}"
            )

    return _parse_llm_response(raw_text)


def _parse_llm_response(raw_text: str) -> tuple[str, str]:
    """Extract improved_instructions and rationale from the LLM JSON response.

    Raises HTTPException(502) when the response is not parseable or missing keys.
    """
    # Strip optional markdown code fences the model sometimes emits despite instructions.
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            f"[_parse_llm_response] Could not parse LLM JSON response | error={exc}"
        )
        raise HTTPException(
            status_code=502,
            detail="prompt_generation_failed: LLM returned a response that could not be parsed as JSON",
        )

    instructions = data.get(_LLM_KEY_INSTRUCTIONS)
    rationale = data.get(_LLM_KEY_RATIONALE)

    if not instructions or not isinstance(instructions, str):
        raise HTTPException(
            status_code=502,
            detail="prompt_generation_failed: LLM response missing 'improved_instructions' field",
        )
    if not rationale or not isinstance(rationale, str):
        raise HTTPException(
            status_code=502,
            detail="prompt_generation_failed: LLM response missing 'rationale' field",
        )

    return instructions.strip(), rationale.strip()


# ── helpers: blob manipulation ────────────────────────────────────────────────


def _extract_instructions(version: ConfigVersion) -> str:
    """Read completion.params.instructions from the source config_blob."""
    blob: dict[str, Any] = version.config_blob or {}
    return blob.get("completion", {}).get("params", {}).get("instructions") or ""
