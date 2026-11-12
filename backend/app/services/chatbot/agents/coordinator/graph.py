"""Coordinator turn: one structured-output call that routes the user to an agent or asks to clarify.

No LangGraph here — a single model call doesn't need a graph.
"""

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.services.chatbot.agents.coordinator.prompts import (
    FALLBACK_CLARIFYING_QUESTION,
    ROUTES_TEMPLATE,
    ROUTING_INSTRUCTION,
)
from app.services.chatbot.agents.onboarding.prompts import EXTRACTION_METHOD
from app.services.chatbot.history import (
    summarize_if_needed,
    to_langchain_messages,
    validate_turn_input,
)
from app.services.chatbot.provider_errors import (
    CHAT_MODEL_ERRORS,
    invoke_structured,
    to_provider_error,
)
from app.services.chatbot.types import ChatbotMessageDict, CoordinatorRoute
from app.services.chatbot.utils import ChatbotModelSettings, get_chat_model

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    intent: str | None = Field(
        default=None,
        description="Exactly one intent name from the list, or null when none clearly matches.",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="Question to ask the user when intent is null; otherwise null.",
    )


@dataclass(frozen=True)
class CoordinatorDecision:
    """Exactly one of `target_node_id` / `clarifying_question` is set."""

    target_node_id: str | None
    clarifying_question: str | None


def _build_routing_prompt(*, system_prompt: str, routes: list[CoordinatorRoute]) -> str:
    route_lines = [
        f"- {route.intent}" + (f": {route.description}" if route.description else "")
        for route in routes
    ]
    return (
        system_prompt
        + ROUTING_INSTRUCTION
        + ROUTES_TEMPLATE.format(routes="\n".join(route_lines))
    )


def _decide_route(
    chat_model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    history_token_limit: int,
) -> RouteDecision:
    conversation_view = summarize_if_needed(
        chat_model, messages, history_token_limit=history_token_limit
    )
    routing_model = chat_model.with_structured_output(
        RouteDecision, method=EXTRACTION_METHOD
    )
    return invoke_structured(
        routing_model, conversation_view, RouteDecision, call_name="Coordinator routing"
    )


def _match_route(
    intent: str | None, routes: list[CoordinatorRoute]
) -> CoordinatorRoute | None:
    if intent is None:
        return None
    wanted = intent.strip().casefold()
    return next((route for route in routes if route.intent.casefold() == wanted), None)


def run_coordinator_turn(
    *,
    system_prompt: str,
    history: list[ChatbotMessageDict],
    user_message: str | None,
    routes: list[CoordinatorRoute],
    model_settings: ChatbotModelSettings,
) -> CoordinatorDecision:
    """Classify the latest user message against `routes`, or return a clarifying question."""
    validate_turn_input(history=history, user_message=user_message)

    messages = to_langchain_messages(
        system_prompt=_build_routing_prompt(system_prompt=system_prompt, routes=routes),
        history=history,
        user_message=user_message,
    )
    try:
        chat_model = get_chat_model(model_settings)
        decision = _decide_route(
            chat_model,
            messages,
            history_token_limit=model_settings.history_token_limit,
        )
    except CHAT_MODEL_ERRORS as e:
        raise to_provider_error(e, model=model_settings.model) from e

    # The opening turn has only the synthetic kickoff message, so it must never hand off.
    if user_message is not None:
        route = _match_route(decision.intent, routes)
        if route is not None:
            logger.info(
                f"[run_coordinator_turn] Routed | intent: {route.intent}, "
                f"target_node_id: {route.target_node_id}"
            )
            return CoordinatorDecision(
                target_node_id=route.target_node_id, clarifying_question=None
            )

    question = decision.clarifying_question
    if question is None or not question.strip():
        question = FALLBACK_CLARIFYING_QUESTION
    logger.info(
        f"[run_coordinator_turn] Clarifying | model_intent: {decision.intent}, "
        f"opening_turn: {user_message is None}"
    )
    return CoordinatorDecision(target_node_id=None, clarifying_question=question)
