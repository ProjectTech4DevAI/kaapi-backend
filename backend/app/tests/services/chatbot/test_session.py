from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import anthropic
import httpx
import pytest
from sqlmodel import Session

from app.crud.chatbot import (
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.crud.config import ConfigVersionCrud
from app.models import (
    ChatbotCompletionStatusEnum,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    Config,
    Project,
)
from app.services.chatbot import session as session_service
from app.services.chatbot.agents.onboarding.types import FieldExtraction
from app.services.chatbot.session import process_chatbot_session_message
from app.tests.utils.chatbot import (
    DEFAULT_SYSTEM_PROMPT,
    ScriptedChatModel,
    add_test_chatbot_messages,
    create_test_chatbot_session,
    extracted_fields,
    patch_chat_model,
)
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
)

WEBHOOK_URL = "https://hooks.example.com/chatbot"


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
def config(db: Session) -> Config:
    return create_test_config(db, config_blob=build_anthropic_text_config_blob())


def _process(chatbot_session: ChatbotSession, db: Session) -> None:
    project = db.get(Project, chatbot_session.project_id)
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


class TestProcessChatbotSessionMessage:
    def test_first_call_stores_greeting_and_posts_success_webhook(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch, ScriptedChatModel(replies=["Hi! What's your name?"])
        )
        chatbot_session = create_test_chatbot_session(db, config=config)

        _process(chatbot_session, db)

        assert _transcript(db, chatbot_session) == [
            ("assistant", "Hi! What's your name?")
        ]
        assert callbacks.calls == [
            (
                WEBHOOK_URL,
                {
                    "success": True,
                    "data": {
                        "session_id": str(chatbot_session.id),
                        "chatbot_ref": chatbot_session.chatbot_ref,
                        "reply": "Hi! What's your name?",
                        "status": "active",
                        "completion_status": "in_progress",
                        "collected_fields": {},
                    },
                },
                None,
            )
        ]

    def test_pending_user_message_is_answered_with_prior_history(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Nice, which city?"])
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
        add_test_chatbot_messages(
            db,
            chatbot_session,
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
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["Which city?"])
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
        add_test_chatbot_messages(
            db,
            chatbot_session,
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
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["should never be used"])
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [
                (ChatbotMessageRoleEnum.USER, "Asha"),
                (ChatbotMessageRoleEnum.ASSISTANT, "Which city?"),
            ],
        )

        _process(chatbot_session, db)

        assert chat_model.calls == []
        assert callbacks.calls == []
        assert _transcript(db, chatbot_session) == [
            ("user", "Asha"),
            ("assistant", "Which city?"),
        ]

    def test_unknown_session_does_nothing(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["should never be used"])
        patch_chat_model(monkeypatch, chat_model)
        project = db.get(Project, config.project_id)
        assert project is not None

        process_chatbot_session_message(
            session_id=uuid4(),
            organization_id=project.organization_id,
            project_id=project.id,
            webhook_url=WEBHOOK_URL,
        )

        assert chat_model.calls == []
        assert callbacks.calls == []

    def test_deleted_config_version_posts_failure_webhook(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["should never be used"])
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
        ConfigVersionCrud(
            session=db, project_id=config.project_id, config_id=config.id
        ).delete_or_raise(1)

        _process(chatbot_session, db)

        assert chat_model.calls == []
        assert _transcript(db, chatbot_session) == []
        assert callbacks.calls == [
            (
                WEBHOOK_URL,
                {
                    "success": False,
                    "error": f"No version found for config '{config.id}'",
                    "data": {"session_id": str(chatbot_session.id)},
                },
                None,
            )
        ]

    def test_non_anthropic_config_posts_failure_webhook(
        self,
        db: Session,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["should never be used"])
        patch_chat_model(monkeypatch, chat_model)
        openai_config = create_test_config(db, use_kaapi_schema=True)
        chatbot_session = create_test_chatbot_session(db, config=openai_config)

        _process(chatbot_session, db)

        assert chat_model.calls == []
        [payload] = callbacks.payloads
        assert payload["success"] is False
        assert "only supports configs" in payload["error"]
        assert payload["data"] == {"session_id": str(chatbot_session.id)}

    def test_provider_error_posts_failure_webhook_and_stores_nothing(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(error=anthropic.APITimeoutError(request=request)),
        )
        chatbot_session = create_test_chatbot_session(db, config=config)
        add_test_chatbot_messages(
            db, chatbot_session, [(ChatbotMessageRoleEnum.USER, "Asha")]
        )

        _process(chatbot_session, db)

        assert _transcript(db, chatbot_session) == [("user", "Asha")]
        [payload] = callbacks.payloads
        assert payload["success"] is False
        assert "(code: APITimeoutError)" in payload["error"]
        assert payload["data"] == {"session_id": str(chatbot_session.id)}

    def test_transcript_without_pending_user_turn_posts_failure_webhook(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(replies=["should never be used"])
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
        # A trailing system row dodges the already-answered check, leaving an
        # assistant turn with no user message after it.
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [
                (ChatbotMessageRoleEnum.ASSISTANT, "Hi!"),
                (ChatbotMessageRoleEnum.SYSTEM, "note"),
            ],
        )

        _process(chatbot_session, db)

        assert chat_model.calls == []
        [payload] = callbacks.payloads
        assert payload["success"] is False
        assert "`message` is required" in payload["error"]


class TestSessionFieldProgress:
    def test_turn_persists_merged_fields_and_completion_and_webhook_reports_them(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["Thanks, all set!"],
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({"city": "Pune"}),
                    missing_fields=[],
                    is_complete=True,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
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
            {
                "success": True,
                "data": {
                    "session_id": str(chatbot_session.id),
                    "chatbot_ref": chatbot_session.chatbot_ref,
                    "reply": "Thanks, all set!",
                    "status": "active",
                    "completion_status": "complete",
                    "collected_fields": {"name": "Asha", "city": "Pune"},
                },
            }
        ]

    def test_extraction_value_error_posts_failure_webhook_and_keeps_progress(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["should never be used"],
            extractions=[ValueError("thinking cannot be disabled for this model")],
        )
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
        add_test_chatbot_messages(
            db, chatbot_session, [(ChatbotMessageRoleEnum.USER, "Asha")]
        )

        _process(chatbot_session, db)

        [payload] = callbacks.payloads
        assert payload["success"] is False
        assert "[KAAPI] Unsupported model configuration" in payload["error"]
        assert "thinking cannot be disabled for this model" in payload["error"]
        assert payload["data"] == {"session_id": str(chatbot_session.id)}
        assert _transcript(db, chatbot_session) == [("user", "Asha")]
        db.expire_all()
        stored = db.get(ChatbotSession, chatbot_session.id)
        assert stored is not None
        assert stored.collected_fields == {}
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS

    def test_complete_session_still_answers_new_messages(
        self,
        db: Session,
        config: Config,
        callbacks: CallbackRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel(
            replies=["You're welcome!"],
            extractions=[
                FieldExtraction(
                    collected_fields=extracted_fields({"name": "Asha", "city": "Pune"}),
                    missing_fields=[],
                    is_complete=True,
                )
            ],
        )
        patch_chat_model(monkeypatch, chat_model)
        chatbot_session = create_test_chatbot_session(db, config=config)
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
