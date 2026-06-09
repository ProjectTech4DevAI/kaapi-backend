from app.models import ResponsesAPIRequest, ResponsesSyncAPIRequest
from app.utils import APIResponse, get_webhook_secret, send_callback


def get_additional_data(request: dict) -> dict:
    """
    Returns extra metadata included in the request payload
    that is not part of the async or sync request models.
    """

    if "assistant_id" in request:
        exclude_keys = set(ResponsesAPIRequest.model_fields.keys())
    else:
        exclude_keys = set(ResponsesSyncAPIRequest.model_fields.keys())
    return {k: v for k, v in request.items() if k not in exclude_keys}


def send_response_callback(
    callback_url: str,
    callback_response: APIResponse,
    request_dict: dict,
    organization_id: int,
    project_id: int,
) -> None:
    """Send a standardized callback response to the provided callback URL."""

    callback_response = callback_response.model_dump()

    webhook_secret = get_webhook_secret(project_id, organization_id)

    send_callback(
        callback_url,
        {
            "success": callback_response.get("success", False),
            "data": {
                **(callback_response.get("data") or {}),
                **get_additional_data(request_dict),
            },
            "error": callback_response.get("error"),
            "metadata": None,
        },
        webhook_secret=webhook_secret,
    )
