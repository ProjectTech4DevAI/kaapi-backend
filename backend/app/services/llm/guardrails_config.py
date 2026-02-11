import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

def fetch_guardrails_config(
    organization_id: int, project_id: int
) -> tuple[list[dict], list[dict]]:
    """
    Fetch guardrail validators and split them into input/output by stage.

    Args:
        organization_id: Organization id
        project_id: Project id

    Retruns:
        List of validators for the given organization id and project id.
    """

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {settings.KAAPI_GUARDRAILS_AUTH}",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                settings.KAAPI_GUARDRAILS_CONFIG_URL,
                params={
                    "organization_id": organization_id,
                    "project_id": project_id,
                },
                headers=headers,
            )

        response.raise_for_status()
        payload = response.json()

        logger.info(f"Added payload: {payload}")

        if not payload.get("success"):
            logger.warning(
                "[fetch_guardrails_config] Guardrails config API returned unsuccessful response. "
                f"organization_id={organization_id}, project_id={project_id}"
            )
            return [], []

        validators = payload.get("data")
        if not isinstance(validators, list):
            logger.warning(
                "[fetch_guardrails_config] Guardrails config response has invalid `data` format. "
                f"organization_id={organization_id}, project_id={project_id}"
            )
            return [], []

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
        logger.warning(
            "[fetch_guardrails_config] Failed to fetch guardrails config. "
            f"organization_id={organization_id}, project_id={project_id}, error={e}"
        )
        return [], []
