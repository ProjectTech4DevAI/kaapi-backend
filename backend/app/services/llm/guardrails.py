import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard
from uuid import UUID

import httpx
from fastapi import HTTPException
from opentelemetry import trace

from app.core.config import settings
from app.models.llm.request import Validator

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


GUARDRAILS_PROXY_TIMEOUT_SECONDS = 30.0


def _guardrails_headers(
    organization_id: int | None, project_id: int | None
) -> dict[str, str]:
    """Guardrails is internal-only: tenant travels in headers set from the
    auth context, never from caller-supplied body/query fields."""
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }
    if organization_id is not None:
        headers["X-ORGANIZATION-ID"] = str(organization_id)
    if project_id is not None:
        headers["X-PROJECT-ID"] = str(project_id)
    return headers


def proxy_guardrails_request(
    method: str,
    path: str,
    *,
    organization_id: int,
    project_id: int,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Forward a management-API call to the guardrails service verbatim.

    No fail-open: these are synchronous CRUD calls, so upstream status codes and
    bodies (including its 422s) are handed back to the caller unchanged.
    """
    url = f"{settings.KAAPI_GUARDRAILS_URL}{path}"
    headers = _guardrails_headers(organization_id, project_id)
    # Unset query params must be omitted, not sent as empty values.
    query = {k: v for k, v in (params or {}).items() if v is not None}

    logger.info(
        f"[proxy_guardrails_request] Forwarding to guardrails | method: {method}, "
        f"url: {url}, organization_id: {organization_id}, project_id: {project_id}"
    )

    try:
        with (
            tracer.start_as_current_span(
                f"guardrails.proxy {method} {path}",
                attributes={
                    "kaapi.organization_id": str(organization_id),
                    "kaapi.project_id": str(project_id),
                },
            ),
            httpx.Client(timeout=GUARDRAILS_PROXY_TIMEOUT_SECONDS) as client,
        ):
            response = client.request(
                method, url, params=query, json=json_body, headers=headers
            )
    except httpx.RequestError as e:
        logger.error(
            f"[proxy_guardrails_request] [KAAPI] Could not reach the guardrails service — "
            f"retry shortly, and contact Kaapi if it persists (code: {type(e).__name__}) | "
            f"method: {method}, url: {url}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=502, detail="Guardrails service unavailable"
        ) from e

    if response.status_code >= 500:
        logger.error(
            f"[proxy_guardrails_request] [GUARDRAILS] Upstream error "
            f"(code: {response.status_code}) | method: {method}, url: {url}"
        )
    elif response.status_code >= 400:
        logger.warning(
            f"[proxy_guardrails_request] [GUARDRAILS] Request rejected "
            f"(code: {response.status_code}) | method: {method}, url: {url}"
        )

    if not response.content:
        return response.status_code, None
    try:
        return response.status_code, response.json()
    except ValueError:
        logger.error(
            f"[proxy_guardrails_request] [GUARDRAILS] Non-JSON response body "
            f"(code: {response.status_code}) | method: {method}, url: {url}"
        )
        raise HTTPException(
            status_code=502, detail="Guardrails service returned an invalid response"
        ) from None


def _is_auth_error(e: Exception) -> TypeGuard[httpx.HTTPStatusError]:
    # 422 included: a missing/invalid tenant header is a backend bug, not a
    # transient outage, so it must not fall open like one.
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (
        401,
        403,
        422,
    )


@dataclass
class GuardrailsOutcome:
    """Result of a single guardrails service call, in domain-agnostic form.

    Callers map this onto their own domain types (QueryParams, BlockResult,
    or a raw text response for the /guardrails endpoint).
    """

    safe_text: str | None

    error: str | None

    bypassed: bool
    """True when the guardrails service was unreachable and we fell back to
    the original text. Callers should treat this as a soft pass."""

    rephrase_needed: bool
    """Input-guardrails-only signal: when True, safe_text is a canned response
    that should be returned directly to the user without invoking the LLM."""

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return bool(self.raw) and not self.bypassed


def apply_guardrails(
    *,
    text: str,
    validators: list[Validator] | None,
    job_id: UUID,
    project_id: int | None,
    organization_id: int | None,
    output_text: str | None = None,
) -> GuardrailsOutcome:
    """Resolve validator configs by ID and run validation against the
    guardrails service. Transport-only — no domain wrappers.

    Used by:
      - /llm/call and /llm/chain (via the apply_input_guardrails /
        apply_output_guardrails adapters in app.services.llm.jobs)
      - /guardrails (dedicated endpoint) directly

    Args:
        text: Primary text to validate. Sent as `input` to the service.
        validators: Validator references (config IDs). When None/empty the
            function short-circuits with a no-op outcome.
        output_text: When provided, sent as `output` to the service for
            output-guardrail validators that compare input/output pairs.
            Also routes the IDs through the output-config fetch path.
    """
    if not validators:
        return GuardrailsOutcome(
            safe_text=text, error=None, bypassed=False, rephrase_needed=False, raw={}
        )

    is_output = output_text is not None
    input_cfgs, output_cfgs = list_validators_config(
        organization_id=organization_id,
        project_id=project_id,
        input_validator_configs=None if is_output else validators,
        output_validator_configs=validators if is_output else None,
    )
    resolved = output_cfgs if is_output else input_cfgs
    if not resolved:
        logger.info(
            f"[apply_guardrails] No validator configs resolved upstream; skipping "
            f"POST /guardrails. job_id={job_id}, requested_ids="
            f"{[str(v.validator_config_id) for v in validators]}, "
            f"is_output={is_output}"
        )
        return GuardrailsOutcome(
            safe_text=text, error=None, bypassed=False, rephrase_needed=False, raw={}
        )

    safe = run_guardrails_validation(
        text,
        resolved,
        job_id,
        project_id,
        organization_id,
        suppress_pass_logs=True,
        output_text=output_text,
    )

    logger.info(
        f"[apply_guardrails] Validation result | success={safe.get('success')}, "
        f"bypassed={safe.get('bypassed', False)}, job_id={job_id}"
    )

    if safe.get("bypassed"):
        return GuardrailsOutcome(
            safe_text=text,
            error=None,
            bypassed=True,
            rephrase_needed=False,
            raw=safe,
        )

    if safe.get("success"):
        data = safe.get("data", {}) or {}
        return GuardrailsOutcome(
            safe_text=data.get("safe_text", text),
            error=None,
            bypassed=False,
            rephrase_needed=bool(data.get("rephrase_needed")),
            raw=safe,
        )

    return GuardrailsOutcome(
        safe_text=None,
        error=safe.get("error"),
        bypassed=False,
        rephrase_needed=False,
        raw=safe,
    )


def run_guardrails_validation(
    input_text: str,
    guardrail_config: Sequence[Validator | dict[str, Any]],
    job_id: UUID,
    project_id: int | None,
    organization_id: int | None,
    suppress_pass_logs: bool = True,
    output_text: str | None = None,
) -> dict[str, Any]:
    """
    Call the Kaapi guardrails service to validate and process input text.

    Args:
        input_text: User query text, maps to payload["input"].
        guardrail_config: List of validator configurations to apply.
        job_id: Unique identifier for the request.
        project_id: Project identifier expected by guardrails API.
        organization_id: Organization identifier expected by guardrails API.
        suppress_pass_logs: Whether to suppress successful validation logs in guardrails service.
        output_text: LLM response text, maps to payload["output"]. Required for validators
            that evaluate input/output pairs.

    Returns:
        JSON response from the guardrails service with validation results.
    """
    validators = [
        validator.model_dump(mode="json")
        if isinstance(validator, Validator)
        else validator
        for validator in guardrail_config
    ]

    payload = {
        "request_id": str(job_id),
        "input": input_text,
        "validators": validators,
    }

    if output_text is not None:
        payload["output"] = output_text

    headers = _guardrails_headers(organization_id, project_id)

    url = f"{settings.KAAPI_GUARDRAILS_URL}/"
    payload_bytes = json.dumps(payload).encode()
    logger.info(
        f"[run_guardrails_validation] POST guardrails | job_id={job_id}, url={url}, "
        f"validators={len(validators)}, input_len={len(input_text)}, "
        f"output_len={len(output_text) if output_text else 0}, "
        f"payload_bytes={len(payload_bytes)}"
    )

    started = time.monotonic()
    try:
        with (
            tracer.start_as_current_span(
                "guardrails.proxy.validate",
                attributes={
                    "kaapi.job_id": str(job_id),
                    "kaapi.organization_id": str(organization_id),
                    "kaapi.project_id": str(project_id),
                    "kaapi.validator_count": len(validators),
                },
            ),
            httpx.Client(timeout=45.0) as client,
        ):
            response = client.post(
                url,
                json=payload,
                params={"suppress_pass_logs": str(suppress_pass_logs).lower()},
                headers=headers,
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                f"[run_guardrails_validation] Response received | job_id={job_id}, "
                f"status={response.status_code}, elapsed_ms={elapsed_ms}, "
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if _is_auth_error(e):
            # Auth failure means a broken deploy (token/IP mismatch), not a
            # transient outage — fail the job instead of silently bypassing.
            logger.error(
                f"[run_guardrails_validation] Guardrails auth failed. "
                f"job_id={job_id}, elapsed_ms={elapsed_ms}, error={e}"
            )
            status_code = e.response.status_code
            return {
                "success": False,
                "bypassed": False,
                # Status only — str(e) embeds the internal service URL and this
                # string is client-visible via job.error_message.
                "error": f"Guardrails service rejected the request (HTTP {status_code})",
            }
        logger.warning(
            f"[run_guardrails_validation] Service unavailable. Bypassing guardrails. "
            f"job_id={job_id}, elapsed_ms={elapsed_ms}, error={e}"
        )

        return {
            "success": False,
            "bypassed": True,
            "data": {
                "safe_text": input_text,
                "rephrase_needed": False,
            },
        }


def list_validators_config(
    organization_id: int | None,
    project_id: int | None,
    input_validator_configs: list[Validator] | None,
    output_validator_configs: list[Validator] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fetch validator configurations by IDs for input and output guardrails.

    Calls:
        GET /validators/configs/?organization_id={organization_id}&project_id={project_id}&ids={uuid}
    """
    input_validator_config_ids = [
        validator_config.validator_config_id
        for validator_config in (input_validator_configs or [])
    ]
    output_validator_config_ids = [
        validator_config.validator_config_id
        for validator_config in (output_validator_configs or [])
    ]

    if not input_validator_config_ids and not output_validator_config_ids:
        return [], []

    headers = _guardrails_headers(organization_id, project_id)

    endpoint = f"{settings.KAAPI_GUARDRAILS_URL}/validators/configs/"

    def _build_params(validator_ids: list[UUID]) -> dict[str, Any]:
        return {
            "ids": [str(validator_config_id) for validator_config_id in validator_ids],
        }

    try:
        with (
            tracer.start_as_current_span(
                "guardrails.proxy.list_validator_configs",
                attributes={
                    "kaapi.organization_id": str(organization_id),
                    "kaapi.project_id": str(project_id),
                },
            ),
            httpx.Client(timeout=10.0) as client,
        ):

            def _fetch_by_ids(validator_ids: list[UUID]) -> list[dict[str, Any]]:
                if not validator_ids:
                    return []

                params = _build_params(validator_ids)
                response = client.get(endpoint, params=params, headers=headers)
                response.raise_for_status()

                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(
                        "Invalid validators response format: expected JSON object."
                    )

                if not payload.get("success", False):
                    raise ValueError(
                        "Validator config fetch failed: `success` is false."
                    )

                validators = payload.get("data", [])
                if not isinstance(validators, list):
                    raise ValueError(
                        "Invalid validators response format: `data` must be a list."
                    )

                return [
                    validator for validator in validators if isinstance(validator, dict)
                ]

            input_guardrails = _fetch_by_ids(input_validator_config_ids)
            output_guardrails = _fetch_by_ids(output_validator_config_ids)
            return input_guardrails, output_guardrails

    except Exception as e:
        if _is_auth_error(e):
            # Propagate so job executors fail the job instead of running
            # without guardrails on a misconfigured token/IP. Sanitized:
            # str(e) embeds the internal service URL and the executors put
            # this message into the client-visible job error.
            logger.error(
                f"[list_validators_config] Guardrails auth failed | "
                f"organization_id={organization_id}, project_id={project_id}, "
                f"endpoint={endpoint}, error={e}"
            )
            raise ValueError(
                f"Guardrails config fetch rejected (HTTP {e.response.status_code})"
            ) from e
        logger.warning(
            "[list_validators_config] Guardrails service unavailable or invalid response. "
            "Proceeding without input/output guardrails. "
            f"input_validator_config_ids={input_validator_config_ids}, output_validator_config_ids={output_validator_config_ids}, "
            f"organization_id={organization_id}, "
            f"project_id={project_id}, endpoint={endpoint}, error={e}"
        )
        return [], []
