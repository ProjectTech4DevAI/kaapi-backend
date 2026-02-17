from typing import Any
from uuid import UUID
import logging

import httpx

from app.core.config import settings
from app.models.llm.request import Validator

logger = logging.getLogger(__name__)


def run_guardrails_validation(
    input_text: str,
    guardrail_config: list[Validator | dict[str, Any]],
    job_id: UUID,
    project_id: int | None,
    organization_id: int | None,
    suppress_pass_logs: bool = True,
) -> dict[str, Any]:
    """
    Call the Kaapi guardrails service to validate and process input text.

    Args:
        input_text: Text to validate and process.
        guardrail_config: List of validator configurations to apply.
        job_id: Unique identifier for the request.
        project_id: Project identifier expected by guardrails API.
        organization_id: Organization identifier expected by guardrails API.
        suppress_pass_logs: Whether to suppress successful validation logs in guardrails service.

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
        "project_id": project_id,
        "organization_id": organization_id,
        "input": input_text,
        "validators": validators,
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{settings.KAAPI_GUARDRAILS_URL}/",
                json=payload,
                params={"suppress_pass_logs": str(suppress_pass_logs).lower()},
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

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }

    endpoint = f"{settings.KAAPI_GUARDRAILS_URL}/validators/configs/"

    def _build_params(validator_ids: list[UUID]) -> dict[str, Any]:
        params = {
            "organization_id": organization_id,
            "project_id": project_id,
            "ids": [str(validator_config_id) for validator_config_id in validator_ids],
        }
        return {key: value for key, value in params.items() if value is not None}

    try:
        with httpx.Client(timeout=10.0) as client:

            def _fetch_by_ids(validator_ids: list[UUID]) -> list[dict[str, Any]]:
                if not validator_ids:
                    return []

                response = client.get(
                    endpoint,
                    params=_build_params(validator_ids),
                    headers=headers,
                )
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
        logger.warning(
            "[list_validators_config] Guardrails service unavailable or invalid response. "
            "Proceeding without input/output guardrails. "
            f"input_validator_config_ids={input_validator_config_ids}, output_validator_config_ids={output_validator_config_ids}, "
            f"organization_id={organization_id}, "
            f"project_id={project_id}, endpoint={endpoint}, error={e}"
        )
        return [], []
