"""OpenAI Decisions API router for the coordinator.

ponytail: raw httpx, the installed openai SDK (2.x) has no `decisions` resource; switch to
`client.decisions.create` once the SDK can move to >=3.26.
"""

import logging
from typing import NamedTuple

import httpx

from app.services.chatbot.agents.coordinator.prompts import (
    DECISIONS_OTHER_DESCRIPTION,
    DECISIONS_ROUTING_INSTRUCTION,
    IN_PROGRESS_TEMPLATE,
)
from app.services.chatbot.exceptions import ChatbotProviderError
from app.services.chatbot.provider_errors import PROVIDER_SERVER_ERROR_STATUS
from app.services.chatbot.types import ChatbotMessageDict, CoordinatorRoute
from app.services.chatbot.utils import FLOW_KEEP_RECENT_MESSAGES

logger = logging.getLogger(__name__)

DECISIONS_URL = "https://api.openai.com/v1/decisions"
DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_TIMEOUT_SECONDS = 10
INTENT_QUESTION_NAME = "intent"
OTHER_INTENT = "other"
REFUSAL_ANSWER_TYPE = "refusal"
# ponytail: guessed with no labeled data; tune from the `[decide_intent]` confidence logs.
ROUTE_CONFIDENCE_THRESHOLD = 0.6
# Routing only needs the recent exchange, so no summarizer call on this path.
DECISIONS_HISTORY_MESSAGES = FLOW_KEEP_RECENT_MESSAGES * 2


class DecisionResult(NamedTuple):
    """`intent` is None when the model chose `other` or was below the confidence threshold."""

    intent: str | None
    confidence: float
    probabilities: dict[str, float]


def _provider_error(
    error_message: str, *, exc: Exception | None = None
) -> ChatbotProviderError:
    logger.error(
        f"[decide_intent] {error_message} | provider=openai, model={DECISIONS_MODEL}",
        exc_info=exc,
    )
    return ChatbotProviderError(error_message)


def decide_intent(
    *,
    api_key: str,
    system_prompt: str,
    routes: list[CoordinatorRoute],
    history: list[ChatbotMessageDict],
    user_message: str,
    in_progress_intent: str | None,
) -> DecisionResult:
    instructions = DECISIONS_ROUTING_INSTRUCTION
    if in_progress_intent is not None:
        instructions += IN_PROGRESS_TEMPLATE.format(intent=in_progress_intent)

    choices: list[dict[str, str]] = []
    for route in routes:
        choice = {"value": route.intent}
        if route.description:
            choice["description"] = route.description
        choices.append(choice)
    choices.append({"value": OTHER_INTENT, "description": DECISIONS_OTHER_DESCRIPTION})

    input_messages: list[ChatbotMessageDict] = [
        {"role": "developer", "content": system_prompt}
    ]
    input_messages.extend(history[-DECISIONS_HISTORY_MESSAGES:])
    input_messages.append({"role": "user", "content": user_message})

    body = {
        "model": DECISIONS_MODEL,
        "input": input_messages,
        "questions": [
            {
                "type": "choice",
                "name": INTENT_QUESTION_NAME,
                "instructions": instructions,
                "choices": choices,
            }
        ],
    }
    try:
        response = httpx.post(
            DECISIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=DECISIONS_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as e:
        raise _provider_error(
            "[KAAPI] Coordinator routing timed out — retry the turn. If persistent, "
            "contact Kaapi. (code: TimeoutException)",
            exc=e,
        ) from e
    except httpx.HTTPError as e:
        raise _provider_error(
            "[KAAPI] Network/DNS issue reaching OpenAI for coordinator routing — check "
            f"connectivity. If persistent, contact Kaapi. (code: {type(e).__name__})",
            exc=e,
        ) from e

    if response.is_error:
        error_message = (
            f"[OPENAI] Coordinator routing failed: {response.text} "
            f"(code: {response.status_code}, request_id: {response.headers.get('x-request-id')})"
        )
        if response.status_code >= PROVIDER_SERVER_ERROR_STATUS:
            raise _provider_error(error_message)
        # 4xx is the caller's fault (bad key, quota), not alert-worthy.
        logger.warning(
            f"[decide_intent] {error_message} | provider=openai, model={DECISIONS_MODEL}"
        )
        raise ChatbotProviderError(error_message)

    try:
        answer = response.json()["answers"][0]
        if answer["type"] == REFUSAL_ANSWER_TYPE:
            raise _provider_error(
                "[OPENAI] Coordinator routing was refused by the model — rephrase the "
                "message. (code: refusal)"
            )
        choice_value = str(answer["choice"])
        confidence = float(answer["confidence"])
        probabilities: dict[str, float] = {}
        for entry in answer.get("probabilities", []):
            probabilities[str(entry["value"])] = float(entry["probability"])
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise _provider_error(
            "[OPENAI] Coordinator routing returned a malformed answer — retry the turn. "
            f"If persistent, contact Kaapi. (code: {type(e).__name__})",
            exc=e,
        ) from e

    intent: str | None = choice_value
    if choice_value == OTHER_INTENT or confidence < ROUTE_CONFIDENCE_THRESHOLD:
        intent = None
    logger.info(
        f"[decide_intent] Decided | choice: {choice_value}, intent: {intent}, "
        f"confidence: {confidence}, probabilities: {probabilities}"
    )
    return DecisionResult(
        intent=intent, confidence=confidence, probabilities=probabilities
    )
