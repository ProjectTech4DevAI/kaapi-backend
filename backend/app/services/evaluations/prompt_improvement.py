"""AI-assisted prompt improvement service.

Loads the evaluation run's stored score traces from object storage, asks Claude
to rewrite the system prompt, and persists the result as a new config_version.
"""

import copy
import json
import logging

import anthropic
from fastapi import HTTPException
from sqlmodel import Session

from app.core.cloud.storage import get_cloud_storage
from app.core.config import settings
from app.core.storage_utils import load_json_from_object_store
from app.crud.config.version import ConfigVersionCrud
from app.crud.evaluations.core import get_evaluation_run_by_id
from app.models.config.config import ConfigTag
from app.models.config.version import (
    ConfigVersionPublic,
    ConfigVersionUpdate,
)
from app.services.llm.providers.claude import ClaudeProvider

logger = logging.getLogger(__name__)

# Room for a full prompt rewrite plus structured JSON wrapper.
_LLM_MAX_TOKENS = 8192

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

    # read_one() re-validates the config exists in this project before returning
    # the version, so a separate ConfigCrud existence check would be redundant.
    version = ConfigVersionCrud(
        session=session,
        config_id=run.config_id,
        project_id=project_id,
        tag=ConfigTag.DEFAULT,
    ).read_one(version_number=run.config_version)
    if version is None:
        raise HTTPException(
            status_code=409,
            detail="source_config_unavailable: the run's config_version is missing or soft-deleted",
        )

    blob = version.config_blob or {}
    params = blob.get("completion", {}).get("params", {}) or {}
    current_instructions = params.get("instructions") or ""

    storage = get_cloud_storage(session=session, project_id=project_id)
    traces = load_json_from_object_store(storage=storage, url=run.score_trace_url)
    if traces is None:
        raise HTTPException(
            status_code=502,
            detail="trace_download_failed: could not retrieve trace file from storage",
        )

    improved_instructions, rationale = _draft_improved_prompt(
        current_instructions=current_instructions,
        config_params=params,
        traces=traces,
    )

    # Derive the new version from the *evaluated* version's blob (not the latest
    # active one) so model, knowledge base, and other params stay apples-to-apples
    # with what was actually scored — only the prompt text changes. The evaluated
    # blob's values therefore win on every key it sets when create_or_raise merges.
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

    logger.info(
        f"[improve_prompt] Done | evaluation_id={evaluation_id} "
        f"new_version_id={new_version.id} version={new_version.version}"
    )

    return ConfigVersionPublic.model_validate(new_version)


def _draft_improved_prompt(
    *,
    current_instructions: str,
    config_params: dict,
    traces: list | dict,
) -> tuple[str, str]:
    """Call Claude with the current prompt + score traces and return
    (improved_instructions, rationale).

    Uses structured outputs so the first text block is guaranteed-valid JSON.
    Raises HTTPException(502) on any LLM failure.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: the platform Anthropic key "
                "(ANTHROPIC_API_KEY) is not configured"
            ),
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
        raise HTTPException(
            status_code=502,
            detail=(
                "prompt_generation_failed: Anthropic authentication failed — "
                "verify the platform API key is valid and not expired"
            ),
        )

    except anthropic.RateLimitError:
        logger.warning(
            "[_draft_improved_prompt] [ANTHROPIC] Rate limit exceeded "
            "(code: 429): Hit Anthropic rate/quota — wait ≥1 min and retry.",
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
            "[_draft_improved_prompt] [KAAPI] Anthropic request timed out "
            "(code: APITimeoutError): retry with a smaller payload.",
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
            "[_draft_improved_prompt] [KAAPI] Anthropic connection failed "
            "(code: APIConnectionError): network or DNS issue reaching Anthropic.",
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
