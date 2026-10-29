from collections.abc import Sequence
from typing import Any
from uuid import UUID

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.chatbot import (
    create_chatbot,
    create_chatbot_flow_version,
    create_chatbot_session,
    create_chatbot_session_message,
    get_latest_chatbot_flow_version,
    get_live_chatbot_flow_version,
    publish_chatbot_flow_version,
)
from app.models import (
    Chatbot,
    ChatbotFlowNodeTypeEnum,
    ChatbotFlowVersion,
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
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
    create_test_project,
    create_test_version,
)
from app.tests.utils.utils import random_lower_string

DEFAULT_SYSTEM_PROMPT = "Collect the user's name and city."
ONBOARDING_NODE_ID = "onboarding-1"
ONBOARDING_NODE_LABEL = "Beneficiary Info Extractor"
# A model other than build_anthropic_text_config_blob's default, so a newer
# config version is distinguishable from version 1.
NEWER_CONFIG_MODEL = "claude-opus-5-5"
WEBHOOK_URL = "https://hooks.example.com/chatbot"
ANTHROPIC_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

CHATBOTS_URL = f"{settings.API_V1_STR}/chatbots"
LATEST_SESSION_URL = f"{CHATBOTS_URL}/sessions/latest"
UNKNOWN_CHATBOT_ID = UUID("00000000-0000-0000-0000-0000000000c1")


def chatbot_url(chatbot_id: UUID) -> str:
    return f"{CHATBOTS_URL}/{chatbot_id}"


def flow_url(chatbot_id: UUID) -> str:
    return f"{chatbot_url(chatbot_id)}/flow"


def publish_url(chatbot_id: UUID) -> str:
    return f"{flow_url(chatbot_id)}/publish"


def sessions_url(chatbot_id: UUID) -> str:
    return f"{chatbot_url(chatbot_id)}/sessions"


def message_url(session_id: UUID) -> str:
    return f"{CHATBOTS_URL}/sessions/{session_id}/message"


def random_contact_ref() -> str:
    return f"contact-{random_lower_string()}"


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
) -> ScriptedChatModel:
    # The Anthropic SDK is the external boundary; everything above it runs for real.
    monkeypatch.setattr(
        onboarding_graph, "get_chat_model", lambda _settings: chat_model
    )
    return chat_model


def create_test_anthropic_config(
    db: Session, *, project_id: int | None = None, **params: Any
) -> Config:
    return create_test_config(
        db,
        project_id=project_id,
        config_blob=build_anthropic_text_config_blob(**params),
    )


def add_test_config_version(
    db: Session, config: Config, *, model: str = NEWER_CONFIG_MODEL, **params: Any
) -> None:
    create_test_version(
        db,
        config_id=config.id,
        project_id=config.project_id,
        config_blob=build_anthropic_text_config_blob(model=model, **params),
    )


def build_flow_node(node_id: str, node_type: str, **data: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": data,
    }


def build_flow_edge(edge_id: str, source: str, target: str) -> dict[str, Any]:
    return {"id": edge_id, "source": source, "target": target}


def build_onboarding_node(
    *,
    config_id: UUID | None,
    config_version: int | None = 1,
    prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    node_id: str = ONBOARDING_NODE_ID,
    node_type: str = ChatbotFlowNodeTypeEnum.ONBOARDING.value,
) -> dict[str, Any]:
    data: dict[str, Any] = {"label": ONBOARDING_NODE_LABEL}
    if config_id is not None:
        data["config_id"] = str(config_id)
    if config_version is not None:
        data["config_version"] = config_version
    if prompt is not None:
        data["prompt"] = prompt
    return build_flow_node(node_id, node_type, **data)


def create_test_chatbot(
    db: Session,
    *,
    project: Project | None = None,
    name: str | None = None,
) -> Chatbot:
    if project is None:
        project = create_test_project(db)
    return create_chatbot(
        db,
        organization_id=project.organization_id,
        project_id=project.id,
        name=name or f"chatbot-{random_lower_string()}",
    )


def create_test_flow_version(
    db: Session,
    *,
    chatbot: Chatbot,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    is_live: bool = False,
) -> ChatbotFlowVersion:
    latest = get_latest_chatbot_flow_version(db, chatbot_id=chatbot.id)
    flow_version = create_chatbot_flow_version(
        db,
        chatbot_id=chatbot.id,
        version=1 if latest is None else latest.version + 1,
        nodes=nodes,
        edges=edges or [],
    )
    if is_live:
        flow_version = publish_chatbot_flow_version(
            db, flow_version=flow_version, nodes=nodes
        )
    return flow_version


def create_test_live_chatbot(
    db: Session,
    *,
    config: Config,
    name: str | None = None,
    prompt: str = DEFAULT_SYSTEM_PROMPT,
    config_version: int = 1,
) -> tuple[Chatbot, ChatbotFlowVersion]:
    """A chatbot in the config's project whose live flow is one onboarding node."""
    project = db.get(Project, config.project_id)
    assert project is not None
    chatbot = create_test_chatbot(db, project=project, name=name)
    flow_version = create_test_flow_version(
        db,
        chatbot=chatbot,
        nodes=[
            build_onboarding_node(
                config_id=config.id, config_version=config_version, prompt=prompt
            )
        ],
        is_live=True,
    )
    return chatbot, flow_version


def create_test_chatbot_session(
    db: Session,
    *,
    chatbot: Chatbot,
    contact_ref: str | None = None,
    flow_version: ChatbotFlowVersion | None = None,
    active_node_id: str = ONBOARDING_NODE_ID,
) -> ChatbotSession:
    if flow_version is None:
        flow_version = get_live_chatbot_flow_version(db, chatbot_id=chatbot.id)
        assert flow_version is not None
    return create_chatbot_session(
        db,
        chatbot_id=chatbot.id,
        flow_version_id=flow_version.id,
        contact_ref=contact_ref or random_contact_ref(),
        active_node_id=active_node_id,
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
                db,
                session_id=chatbot_session.id,
                role=role,
                content=content,
                node_id=chatbot_session.active_node_id,
            )
        )
    return stored


def list_stored_flow_versions(
    db: Session, chatbot: Chatbot
) -> list[ChatbotFlowVersion]:
    db.expire_all()
    return list(
        db.exec(
            select(ChatbotFlowVersion)
            .where(ChatbotFlowVersion.chatbot_id == chatbot.id)
            .order_by(ChatbotFlowVersion.version)
        ).all()
    )


def list_stored_sessions_for_contact(
    db: Session, contact_ref: str
) -> list[ChatbotSession]:
    db.expire_all()
    return list(
        db.exec(
            select(ChatbotSession).where(ChatbotSession.contact_ref == contact_ref)
        ).all()
    )
