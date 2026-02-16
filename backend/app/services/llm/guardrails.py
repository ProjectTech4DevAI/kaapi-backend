from typing import Any
from uuid import UUID
import logging

import httpx

from app.core.config import settings
from app.models.llm.request import Validator

logger = logging.getLogger(__name__)


def run_guardrails_validation(
    input_text: str, guardrail_config: list[dict[str, Any] | Validator], job_id: UUID
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
    validators_payload = [
        validator.model_dump() if isinstance(validator, Validator) else validator
        for validator in guardrail_config
    ]

    payload = {
        "request_id": str(job_id),
        "input": input_text,
        "validators": validators_payload,
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


def get_validators_config(
    validator_configs: list[Validator] | None,
    organization_id: int | None,
    project_id: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fetch validator configurations from batch payload and split by stage.

    Calls:
        POST /validators/configs/batch/fetch?organization_id={organization_id}&project_id={project_id}
    """
    if not validator_configs:
        return [], []

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
        "Content-Type": "application/json",
    }

    endpoint = f"{settings.KAAPI_GUARDRAILS_URL}/validators/configs/batch/fetch"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                endpoint,
                params={
                    "organization_id": organization_id,
                    "project_id": project_id,
                },
                json=[validator.model_dump() for validator in validator_configs],
                headers=headers,
            )
            response.raise_for_status()

            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(
                    "Invalid validators response format: expected JSON object."
                )

            if not payload.get("success", False):
                raise ValueError("Validator config fetch failed: `success` is false.")

            validators = payload.get("data", [])
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
            f"validator_configs={validator_configs}, organization_id={organization_id}, project_id={project_id}, "
            f"endpoint={endpoint}, error={e}"
        )
        raise
