from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field
from sqlmodel import Session

from app.crud.chatbot import create_chatbot_session, create_chatbot_session_message
from app.models import (
    DEFAULT_CHATBOT_NAME,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionMessage,
    Config,
    Project,
)
from app.services.chatbot.agents.onboarding import graph as onboarding_graph
from app.services.chatbot.agents.onboarding.types import (
    ExtractedField,
    FieldExtraction,
)
from app.tests.utils.utils import random_lower_string

DEFAULT_SYSTEM_PROMPT = "Collect the user's name and city."


def extracted_fields(fields: dict[str, str]) -> list[ExtractedField]:
    return [ExtractedField(name=name, value=value) for name, value in fields.items()]


NOTHING_EXTRACTED = FieldExtraction(
    collected_fields=[], missing_fields=[], is_complete=False
)


class ScriptedChatModel(BaseChatModel):
    """Stands in for ChatAnthropic: replies from a script and records each call."""

    replies: list[str] = Field(default_factory=list)
    token_count: int = 0
    error: Exception | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    # Each entry is a FieldExtraction to return, an exception to raise, or any
    # other object to simulate a wrongly-typed structured output.
    extractions: list[Any] = Field(default_factory=list)
    extraction_calls: list[list[BaseMessage]] = Field(default_factory=list)
    structured_output_methods: list[str | None] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        reply = AIMessage(content=self.replies.pop(0))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def get_num_tokens_from_messages(
        self, messages: list[BaseMessage], tools: Sequence[Any] | None = None
    ) -> int:
        return self.token_count

    def with_structured_output(
        self, schema: Any, *, method: str | None = None, **kwargs: Any
    ) -> Runnable[Any, Any]:
        self.structured_output_methods.append(method)

        def _extract(messages: list[BaseMessage]) -> Any:
            self.extraction_calls.append(list(messages))
            outcome = self.extractions.pop(0) if self.extractions else NOTHING_EXTRACTED
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return RunnableLambda(_extract)


def patch_chat_model(
    monkeypatch: pytest.MonkeyPatch, chat_model: ScriptedChatModel
) -> None:
    # The Anthropic SDK is the external boundary; everything above it runs for real.
    monkeypatch.setattr(
        onboarding_graph, "get_chat_model", lambda _settings: chat_model
    )


def create_test_chatbot_session(
    db: Session,
    *,
    config: Config,
    name: str = DEFAULT_CHATBOT_NAME,
    contact_ref: str | None = None,
    config_version: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> ChatbotSession:
    project = db.get(Project, config.project_id)
    assert project is not None
    return create_chatbot_session(
        db,
        organization_id=project.organization_id,
        project_id=project.id,
        name=name,
        contact_ref=contact_ref or f"contact-{random_lower_string()}",
        config_id=config.id,
        config_version=config_version,
        system_prompt=system_prompt,
    )


def add_test_chatbot_messages(
    db: Session,
    chatbot_session: ChatbotSession,
    turns: list[tuple[ChatbotMessageRoleEnum, str]],
) -> list[ChatbotSessionMessage]:
    stored: list[ChatbotSessionMessage] = []
    for role, content in turns:
        stored.append(
            create_chatbot_session_message(
                db, session_id=chatbot_session.id, role=role, content=content
            )
        )
    return stored
