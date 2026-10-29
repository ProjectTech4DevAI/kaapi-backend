from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import anthropic
import pytest
from sqlmodel import Session

from app.crud.chatbot import (
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.crud.config import ConfigVersionCrud
from app.models import (
    Chatbot,
    ChatbotCompletionStatusEnum,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    Config,
    Project,
)
from app.services.chatbot import session as session_service
from app.services.chatbot.agents.onboarding import graph as onboarding_graph
from app.services.chatbot.agents.onboarding.types import FieldExtraction
from app.services.chatbot.session import (
    SUPERSEDED_SESSION_ERROR,
    process_chatbot_session_message,
)
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import (
    ANTHROPIC_REQUEST,
    DEFAULT_SYSTEM_PROMPT,
    ONBOARDING_NODE_ID,
    WEBHOOK_URL,
    ScriptedChatModel,
    add_test_chatbot_messages,
    add_test_config_version,
    build_onboarding_node,
    create_test_chatbot,
    create_test_chatbot_session,
    create_test_flow_version,
    create_test_live_chatbot,
    extracted_fields,
    patch_chat_model,
)
from app.tests.utils.test_data import create_test_config

ChatTurns = list[tuple[ChatbotMessageRoleEnum, str]]


class _NonClosingSession:
    """The worker opens `Session(engine)`, which can't see this test's
    uncommitted rows; hand it the transactional test session instead."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(self, *_exc: object) -> bool:
        return False


@dataclass
class CallbackRecorder:
    calls: list[tuple[str, dict[str, Any], str | None]] = field(default_factory=list)

    def __call__(
        self, url: str, payload: dict[str, Any], secret: str | None = None
    ) -> bool:
        self.calls.append((url, payload, secret))
        return True

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [payload for _url, payload, _secret in self.calls]


@pytest.fixture
def callbacks(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> Iterator[CallbackRecorder]:
    recorder = CallbackRecorder()
    monkeypatch.setattr(
        session_service, "Session", lambda _engine: _NonClosingSession(db)
    )
    # The webhook POST is the external boundary.
    monkeypatch.setattr(session_service, "send_callback", recorder)
    yield recorder


@pytest.fixture
def chatbot(db: Session, config: Config) -> Chatbot:
    return create_test_live_chatbot(db, config=config)[0]


@pytest.fixture
def unused_chat_model(monkeypatch: pytest.MonkeyPatch) -> ScriptedChatModel:
    """A patched model for turns that must fail or skip before any LLM call."""
    return patch_chat_model(
        monkeypatch, ScriptedChatModel(replies=["should never be used"])
    )


def _process(chatbot_session: ChatbotSession, db: Session) -> None:
    chatbot = db.get(Chatbot, chatbot_session.chatbot_id)
    assert chatbot is not None
    project = db.get(Project, chatbot.project_id)
    assert project is not None
    process_chatbot_session_message(
        session_id=chatbot_session.id,
        organization_id=project.organization_id,
        project_id=project.id,
        webhook_url=WEBHOOK_URL,
    )


def _transcript(db: Session, chatbot_session: ChatbotSession) -> list[tuple[str, str]]:
    return [
        (m.role.value, m.content)
        for m in list_chatbot_session_messages(db, session_id=chatbot_session.id)
    ]


def _session_with_turns(
    db: Session, chatbot: Chatbot, turns: ChatTurns
) -> ChatbotSession:
    chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
    add_test_chatbot_messages(db, chatbot_session, turns)
    return chatbot_session


def _success_webhook(
    chatbot_session: ChatbotSession,
    chatbot: Chatbot,
    *,
    reply: str,
    completion_status: str = "in_progress",
    collected_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "data": {
            "session_id": str(chatbot_session.id),
            "chatbot_id": str(chatbot.id),
            "node_id": ONBOARDING_NODE_ID,
            "reply": reply,
            "status": "active",
            "completion_status": completion_status,
            "collected_fields": collected_fields or {},
        },
    }


def _failure_webhook(chatbot_session: ChatbotSession, error: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": error,
        "data": {"session_id": str(chatbot_session.id)},
    }


def _assert_single_failure_webhook(
    callbacks: CallbackRecorder,
    chatbot_session: ChatbotSession,
    *error_fragments: str,
) -> None:
    [payload] = callbacks.payloads
    assert payload["success"] is False
    for fragment in error_fragments:
        assert fragment in payload["error"]
    assert payload["data"] == {"session_id": str(chatbot_session.id)}


class TestProcessChatbotSessionMessage:
    def test_first_call_stores_greeting_and_posts_success_webhook(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch, ScriptedChatModel(replies=["Hi! What's your name?"])
        )
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)

        _process(chatbot_session, db)

        assert _transcript(db, chatbot_session) == [
            ("assistant", "Hi! What's your name?")
        ]
        assert callbacks.calls == [
            (
                WEBHOOK_URL,
                _success_webhook(
                    chatbot_session, chatbot, reply="Hi! What's your name?"
                ),
                None,
            )
        ]

    def test_pending_user_message_is_answered_with_prior_history(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = patch_chat_model(
            monkeypatch, ScriptedChatModel(replies=["Nice, which city?"])
        )
        chatbot_session = _session_with_turns(
            db,
            chatbot,
            [
                (ChatbotMessageRoleEnum.ASSISTANT, "Hi! What's your name?"),
                (ChatbotMessageRoleEnum.USER, "Asha"),
            ],
        )

        _process(chatbot_session, db)

        assert [m.content for m in chat_model.calls[0]] == [
            DEFAULT_SYSTEM_PROMPT,
            "Begin the conversation.",
            "Hi! What's your name?",
            "Asha",
        ]
        assert _transcript(db, chatbot_session) == [
            ("assistant", "Hi! What's your name?"),
            ("user", "Asha"),
            ("assistant", "Nice, which city?"),
        ]
        assert callbacks.payloads[0]["data"]["reply"] == "Nice, which city?"

    def test_stored_system_rows_are_not_sent_as_history(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = patch_chat_model(
            monkeypatch, ScriptedChatModel(replies=["Which city?"])
        )
        chatbot_session = _session_with_turns(
            db,
            chatbot,
            [
                (ChatbotMessageRoleEnum.SYSTEM, "stale stored prompt"),
                (ChatbotMessageRoleEnum.USER, "I'm Asha"),
            ],
        )

        _process(chatbot_session, db)

        sent = [m.content for m in chat_model.calls[0]]
        assert "stale stored prompt" not in sent
        assert sent == [DEFAULT_SYSTEM_PROMPT, "I'm Asha"]
        assert callbacks.payloads[0]["success"] is True

    def test_already_answered_turn_is_skipped(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        chatbot_session = _session_with_turns(
            db,
            chatbot,
            [
                (ChatbotMessageRoleEnum.USER, "Asha"),
                (ChatbotMessageRoleEnum.ASSISTANT, "Which city?"),
            ],
        )

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        assert callbacks.calls == []
        assert _transcript(db, chatbot_session) == [
            ("user", "Asha"),
            ("assistant", "Which city?"),
        ]

    def test_unknown_session_does_nothing(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        project = db.get(Project, chatbot.project_id)
        assert project is not None

        process_chatbot_session_message(
            session_id=uuid4(),
            organization_id=project.organization_id,
            project_id=project.id,
            webhook_url=WEBHOOK_URL,
        )

        assert unused_chat_model.calls == []
        assert callbacks.calls == []

    def test_deleted_config_version_posts_failure_webhook(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        ConfigVersionCrud(
            session=db, project_id=config.project_id, config_id=config.id
        ).delete_or_raise(1)

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        assert _transcript(db, chatbot_session) == []
        assert callbacks.calls == [
            (
                WEBHOOK_URL,
                _failure_webhook(
                    chatbot_session,
                    f"Version with number '1' not found for config '{config.id}'",
                ),
                None,
            )
        ]

    def test_turn_runs_the_pinned_version_after_a_newer_one_is_saved(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Hi! What's your name?"])
        used_models: list[str] = []

        def _get_chat_model(model_settings: ChatbotModelSettings) -> ScriptedChatModel:
            used_models.append(model_settings.model)
            return chat_model

        monkeypatch.setattr(onboarding_graph, "get_chat_model", _get_chat_model)
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        add_test_config_version(db, config)

        _process(chatbot_session, db)

        assert used_models == ["claude-sonnet-4-6"]
        assert callbacks.calls[0][1]["success"] is True

    def test_non_anthropic_config_posts_failure_webhook(
        self,
        db: Session,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        openai_config = create_test_config(db, use_kaapi_schema=True)
        # Publish would reject this config; the worker must still guard against it.
        openai_chatbot, _ = create_test_live_chatbot(db, config=openai_config)
        chatbot_session = create_test_chatbot_session(db, chatbot=openai_chatbot)

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        _assert_single_failure_webhook(
            callbacks, chatbot_session, "only supports configs"
        )

    def test_provider_error_posts_failure_webhook_and_stores_nothing(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(
                error=anthropic.APITimeoutError(request=ANTHROPIC_REQUEST)
            ),
        )
        chatbot_session = _session_with_turns(
            db, chatbot, [(ChatbotMessageRoleEnum.USER, "Asha")]
        )

        _process(chatbot_session, db)

        assert _transcript(db, chatbot_session) == [("user", "Asha")]
        _assert_single_failure_webhook(
            callbacks, chatbot_session, "(code: APITimeoutError)"
        )

    def test_transcript_without_pending_user_turn_posts_failure_webhook(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        # A trailing system row dodges the already-answered check, leaving an
        # assistant turn with no user message after it.
        chatbot_session = _session_with_turns(
            db,
            chatbot,
            [
                (ChatbotMessageRoleEnum.ASSISTANT, "Hi!"),
                (ChatbotMessageRoleEnum.SYSTEM, "note"),
            ],
        )

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        [payload] = callbacks.payloads
        assert payload["success"] is False
        assert "`message` is required" in payload["error"]


class TestSessionFieldProgress:
    def test_turn_persists_merged_fields_and_completion_and_webhook_reports_them(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(
                replies=["Thanks, all set!"],
                extractions=[
                    FieldExtraction(
                        collected_fields=extracted_fields({"city": "Pune"}),
                        missing_fields=[],
                        is_complete=True,
                    )
                ],
            ),
        )
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha"},
            completion_status=ChatbotCompletionStatusEnum.IN_PROGRESS,
        )
        updated_before_turn = chatbot_session.updated_at
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [
                (ChatbotMessageRoleEnum.ASSISTANT, "Which city?"),
                (ChatbotMessageRoleEnum.USER, "Pune"),
            ],
        )

        _process(chatbot_session, db)

        db.expire_all()
        stored = db.get(ChatbotSession, chatbot_session.id)
        assert stored is not None
        # "name" only survives if the stored prior map was handed to the engine.
        assert stored.collected_fields == {"name": "Asha", "city": "Pune"}
        assert stored.completion_status == ChatbotCompletionStatusEnum.COMPLETE
        assert stored.updated_at > updated_before_turn
        assert callbacks.payloads == [
            _success_webhook(
                chatbot_session,
                chatbot,
                reply="Thanks, all set!",
                completion_status="complete",
                collected_fields={"name": "Asha", "city": "Pune"},
            )
        ]

    def test_extraction_value_error_posts_failure_webhook_and_keeps_progress(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(
                replies=["should never be used"],
                extractions=[ValueError("thinking cannot be disabled for this model")],
            ),
        )
        chatbot_session = _session_with_turns(
            db, chatbot, [(ChatbotMessageRoleEnum.USER, "Asha")]
        )

        _process(chatbot_session, db)

        _assert_single_failure_webhook(
            callbacks,
            chatbot_session,
            "[KAAPI] Unsupported model configuration",
            "thinking cannot be disabled for this model",
        )
        assert _transcript(db, chatbot_session) == [("user", "Asha")]
        db.expire_all()
        stored = db.get(ChatbotSession, chatbot_session.id)
        assert stored is not None
        assert stored.collected_fields == {}
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS

    def test_complete_session_still_answers_new_messages(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(
                replies=["You're welcome!"],
                extractions=[
                    FieldExtraction(
                        collected_fields=extracted_fields(
                            {"name": "Asha", "city": "Pune"}
                        ),
                        missing_fields=[],
                        is_complete=True,
                    )
                ],
            ),
        )
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha", "city": "Pune"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [
                (ChatbotMessageRoleEnum.ASSISTANT, "All set!"),
                (ChatbotMessageRoleEnum.USER, "Thanks"),
            ],
        )

        _process(chatbot_session, db)

        assert _transcript(db, chatbot_session) == [
            ("assistant", "All set!"),
            ("user", "Thanks"),
            ("assistant", "You're welcome!"),
        ]
        [payload] = callbacks.payloads
        assert payload["success"] is True
        assert payload["data"]["completion_status"] == "complete"


def _session_on_live_flow(
    db: Session,
    config: Config,
    *,
    nodes: list[dict[str, Any]],
    active_node_id: str,
) -> ChatbotSession:
    project = db.get(Project, config.project_id)
    assert project is not None
    chatbot = create_test_chatbot(db, project=project)
    flow_version = create_test_flow_version(
        db, chatbot=chatbot, nodes=nodes, is_live=True
    )
    return create_test_chatbot_session(
        db, chatbot=chatbot, flow_version=flow_version, active_node_id=active_node_id
    )


class TestProcessChatbotSessionMessageFlowNode:
    def test_reply_is_stored_against_the_active_node(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(monkeypatch, ScriptedChatModel(replies=["Which city?"]))
        chatbot_session = _session_with_turns(
            db, chatbot, [(ChatbotMessageRoleEnum.USER, "Asha")]
        )

        _process(chatbot_session, db)

        stored = list_chatbot_session_messages(db, session_id=chatbot_session.id)
        assert [(m.role.value, m.node_id) for m in stored] == [
            ("user", ONBOARDING_NODE_ID),
            ("assistant", ONBOARDING_NODE_ID),
        ]

    def test_pinned_flow_prompt_is_used_after_a_newer_flow_is_published(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = patch_chat_model(
            monkeypatch, ScriptedChatModel(replies=["Hi! What's your name?"])
        )
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        create_test_flow_version(
            db,
            chatbot=chatbot,
            nodes=[
                build_onboarding_node(
                    config_id=config.id, prompt="Collect the student's grade."
                )
            ],
            is_live=True,
        )

        _process(chatbot_session, db)

        assert chat_model.calls[0][0].content == DEFAULT_SYSTEM_PROMPT
        assert callbacks.payloads[0]["success"] is True

    def test_inactive_session_posts_superseded_failure_webhook(
        self,
        db: Session,
        chatbot: Chatbot,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        chatbot_session = _session_with_turns(
            db, chatbot, [(ChatbotMessageRoleEnum.USER, "Asha")]
        )
        create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=chatbot_session.contact_ref
        )

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        assert callbacks.payloads == [
            _failure_webhook(chatbot_session, SUPERSEDED_SESSION_ERROR)
        ]
        assert _transcript(db, chatbot_session) == [("user", "Asha")]

    def test_active_node_missing_from_flow_posts_failure_webhook(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        chatbot_session = _session_on_live_flow(
            db,
            config,
            nodes=[build_onboarding_node(config_id=config.id)],
            active_node_id="ghost",
        )

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        assert callbacks.payloads == [
            _failure_webhook(
                chatbot_session,
                "[KAAPI] Active node 'ghost' is not in the session's flow "
                "version; start a new session.",
            )
        ]

    def test_non_onboarding_active_node_posts_failure_webhook(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
    ) -> None:
        coordinator = build_onboarding_node(
            config_id=config.id, node_id="router", node_type="coordinator"
        )
        chatbot_session = _session_on_live_flow(
            db, config, nodes=[coordinator], active_node_id="router"
        )

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        [payload] = callbacks.payloads
        assert payload["success"] is False
        assert "'router' has type 'coordinator'" in payload["error"]
        assert "only 'onboarding' nodes can run" in payload["error"]

    @pytest.mark.parametrize(
        "node_overrides",
        [{"prompt": None}, {"config_id": None}],
        ids=["missing-prompt", "missing-config"],
    )
    def test_node_without_prompt_or_config_posts_failure_webhook(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        unused_chat_model: ScriptedChatModel,
        node_overrides: dict[str, Any],
    ) -> None:
        node_kwargs: dict[str, Any] = {"config_id": config.id, **node_overrides}
        chatbot_session = _session_on_live_flow(
            db,
            config,
            nodes=[build_onboarding_node(**node_kwargs)],
            active_node_id=ONBOARDING_NODE_ID,
        )

        _process(chatbot_session, db)

        assert unused_chat_model.calls == []
        _assert_single_failure_webhook(
            callbacks, chatbot_session, "missing data.prompt or data.config_id"
        )
