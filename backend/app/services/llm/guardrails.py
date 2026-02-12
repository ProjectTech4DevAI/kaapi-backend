from typing import Any
from uuid import UUID
import logging

import httpx

from app.core.config import settings
from app.models.llm.request import GuardrailsConfig

logger = logging.getLogger(__name__)


def run_guardrails_validation(
    input_text: str, guardrail_config: list[dict], job_id: UUID
) -> dict[str, Any]:
    """
    Call the Kaapi guardrails service to validate and process input text.

    Args:
        input_text: Text to validate and process.
        guardrail_config: List of validator configurations to apply.
        job_id: Unique identifier for the request.

    Returns:
        JSON response from the guardrails service with validation results.
    """
    payload = {
        "request_id": str(job_id),
        "input": input_text,
        "validators": guardrail_config,
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                settings.KAAPI_GUARDRAILS_URL,
                json=payload,
                headers=headers,
            )

            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(
            f"[run_guardrails_validation] Service unavailable. Bypassing guardrails. job_id={job_id}. error={e}"
        )

        return {
            "success": False,
            "bypassed": True,
            "data": {
                "safe_text": input_text,
                "rephrase_needed": False,
            },
        }


def create_validators_batch(
    validators: list[dict[str, Any]],
    config_id: UUID | None,
    organization_id: int | None,
    project_id: int | None,
) -> list[dict[str, Any]]:
    """
    Batch create validator configs via Kaapi Guardrails service.

    Args:
        validators: List of validator creation payloads
        config_id: Optional config UUID associated with this batch

    Returns:
        List of created validator objects (includes UUIDs)
    """

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }

    try:
        payload: dict[str, Any] | list[dict]

        if config_id is None:
            raise ValueError("config_id must be provided")

        payload = {
            "config_id": str(config_id) if config_id is not None else None,
            "validators": validators,
        }

        logger.info(
            "[create_validators_batch] Requesting validator batch creation. "
            f"config_id={config_id}, organization_id={organization_id}, "
            f"project_id={project_id}, validators_count={len(validators)}"
        )

        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{settings.KAAPI_GUARDRAILS_URL}/validators/configs/batch",
                params={
                    "organization_id": organization_id,
                    "project_id": project_id,
                },
                json=payload,
                headers=headers,
            )

            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(
                    "Invalid response format from guardrails service: expected object."
                )

            validators_data = data.get("data")
            if not isinstance(validators_data, list):
                raise ValueError(
                    "Invalid response format from guardrails service: `data` must be a list."
                )

            return validators_data

    except Exception as e:
        logger.error(
            f"[create_validators_batch] Failed to create validators. error={e}"
        )
        raise


def get_validators_config(
    config_id: UUID | str,
    organization_id: int | None,
    project_id: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fetch validator configuration for a specific config id and split by stage.

    Calls:
        GET /validators/configs/{config_id}?organization_id={organization_id}&project_id={project_id}
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
    }

    endpoint = f"{settings.KAAPI_GUARDRAILS_URL}/validators/configs/{config_id}"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                endpoint,
                params={
                    "organization_id": organization_id,
                    "project_id": project_id,
                },
                headers=headers,
            )
            response.raise_for_status()

            payload = response.json()
            validators = payload.get("data", []) if isinstance(payload, dict) else []

            if not isinstance(validators, list):
                raise ValueError(
                    "Invalid validators response format: `data` must be a list."
                )

            input_guardrails = [
                validator
                for validator in validators
                if isinstance(validator, dict)
                and str(validator.get("stage", "")).lower() == "input"
            ]
            output_guardrails = [
                validator
                for validator in validators
                if isinstance(validator, dict)
                and str(validator.get("stage", "")).lower() == "output"
            ]

            return input_guardrails, output_guardrails

    except Exception as e:
        logger.error(
            "[get_validators_config] Failed to fetch validator config. "
            f"config_id={config_id}, organization_id={organization_id}, project_id={project_id}, error={e}"
        )
        raise


def build_staged_validators(
    guardrails: GuardrailsConfig | None,
) -> list[dict[str, Any]]:
    validators: list[dict[str, Any]] = []
    if guardrails is None:
        return validators

    for validator in guardrails.input or []:
        validators.append({"stage": "input", **validator})
    for validator in guardrails.output or []:
        validators.append({"stage": "output", **validator})

    return validators


def create_guardrails_validators_if_present(
    guardrails: GuardrailsConfig | None,
    guardrails_config_id: UUID,
    organization_id: int | None,
    project_id: int | None,
) -> None:
    validators = build_staged_validators(guardrails)
    if not validators:
        return

    create_validators_batch(
        validators=validators,
        config_id=guardrails_config_id,
        organization_id=organization_id,
        project_id=project_id,
    )
