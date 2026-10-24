import re
import socket
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.routes.chatbot import session as session_routes
from app.core.config import settings
from app.crud.chatbot import (
    get_chatbot_session,
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.models import (
    DEFAULT_CHATBOT_NAME,
    ChatbotCompletionStatusEnum,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionStatusEnum,
    Config,
)
from app.services.chatbot import chatbot_ref as chatbot_ref_module
from app.services.chatbot.chatbot_ref import (
    CHATBOT_REF_PREFIX,
    CHATBOT_REF_SUFFIX_LENGTH,
)
from app.services.chatbot.exceptions import (
    CONCURRENT_SESSION_CONFLICT_MESSAGE,
    ChatbotSessionConflictError,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import (
    add_test_chatbot_messages,
    create_test_chatbot_session,
)
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
)
from app.tests.utils.utils import random_lower_string

SESSIONS_URL = f"{settings.API_V1_STR}/chatbot/sessions"
WEBHOOK_URL = "https://hooks.example.com/chatbot"
PUBLIC_IP = "93.184.216.34"
SYSTEM_PROMPT = "Collect the user's name and city."


def _message_url(session_id: Any) -> str:
    return f"{SESSIONS_URL}/{session_id}/message"


def _sessions_for_contact(db: Session, contact_ref: str) -> list[ChatbotSession]:
    db.expire_all()
    return list(
        db.exec(
            select(ChatbotSession).where(ChatbotSession.contact_ref == contact_ref)
        ).all()
    )


def _supersede(db: Session, chatbot_session: ChatbotSession) -> ChatbotSession:
    config = db.get(Config, chatbot_session.config_id)
    assert config is not None
    return create_test_chatbot_session(
        db,
        config=config,
        contact_ref=chatbot_session.contact_ref,
        chatbot_ref=chatbot_session.chatbot_ref,
    )


@pytest.fixture
def headers(user_api_key: TestAuthContext) -> dict[str, str]:
    return {"X-API-KEY": user_api_key.key}


@pytest.fixture
def anthropic_config(db: Session, user_api_key: TestAuthContext) -> Config:
    return create_test_config(
        db,
        project_id=user_api_key.project_id,
        config_blob=build_anthropic_text_config_blob(),
    )


@pytest.fixture
def chatbot_session(db: Session, anthropic_config: Config) -> ChatbotSession:
    return create_test_chatbot_session(db, config=anthropic_config)


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    # The Celery broker is the external boundary; patched where the route imported it.
    monkeypatch.setattr(
        session_routes,
        "start_chatbot_session_message",
        lambda **kwargs: calls.append(kwargs) or "task-id",
    )
    # validate_callback_url resolves DNS; pin it to a public address.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))
        ],
    )
    return calls


class TestCreateChatbotSession:
    def test_creates_session_in_callers_project(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        user_api_key: TestAuthContext,
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"

        resp = client.post(
            SESSIONS_URL,
            headers=headers,
            json={
                "name": "Onboarding Bot",
                "config_id": str(anthropic_config.id),
                "config_version": 1,
                "contact_ref": contact_ref,
                "chatbot_ref": "chatbot-onboarding",
                "system_prompt": SYSTEM_PROMPT,
            },
        )

        assert resp.status_code == 201
        session_id = resp.json()["data"]["session_id"]
        stored = get_chatbot_session(
            db,
            session_id=session_id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        assert stored is not None
        assert stored.name == "Onboarding Bot"
        assert stored.contact_ref == contact_ref
        assert stored.config_id == anthropic_config.id
        assert stored.config_version == 1
        assert stored.system_prompt == SYSTEM_PROMPT

    def test_omitted_name_uses_default(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        user_api_key: TestAuthContext,
        anthropic_config: Config,
    ) -> None:
        resp = client.post(
            SESSIONS_URL,
            headers=headers,
            json={
                "config_id": str(anthropic_config.id),
                "contact_ref": f"contact-{random_lower_string()}",
                "chatbot_ref": f"chatbot-{random_lower_string()}",
                "system_prompt": SYSTEM_PROMPT,
            },
        )

        assert resp.status_code == 201
        stored = get_chatbot_session(
            db,
            session_id=resp.json()["data"]["session_id"],
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        assert stored is not None
        assert stored.name == DEFAULT_CHATBOT_NAME

    def test_unknown_config_returns_404(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        resp = client.post(
            SESSIONS_URL,
            headers=headers,
            json={
                "config_id": str(uuid4()),
                "contact_ref": "contact-1",
                "chatbot_ref": "chatbot-1",
                "system_prompt": SYSTEM_PROMPT,
            },
        )

        assert resp.status_code == 404

    def test_non_anthropic_config_returns_422(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        user_api_key: TestAuthContext,
    ) -> None:
        openai_config = create_test_config(
            db, project_id=user_api_key.project_id, use_kaapi_schema=True
        )

        resp = client.post(
            SESSIONS_URL,
            headers=headers,
            json={
                "config_id": str(openai_config.id),
                "contact_ref": "contact-1",
                "chatbot_ref": "chatbot-1",
                "system_prompt": SYSTEM_PROMPT,
            },
        )

        assert resp.status_code == 422
        assert "only supports configs" in resp.json()["error"]


class TestCreateChatbotSessionChatbotRef:
    def _create(
        self,
        client: TestClient,
        headers: dict[str, str],
        config: Config,
        contact_ref: str,
        **extra: Any,
    ) -> Any:
        return client.post(
            SESSIONS_URL,
            headers=headers,
            json={
                "config_id": str(config.id),
                "contact_ref": contact_ref,
                "system_prompt": SYSTEM_PROMPT,
                **extra,
            },
        )

    def test_omitted_chatbot_ref_is_generated_and_stored(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"

        resp = self._create(client, headers, anthropic_config, contact_ref)

        assert resp.status_code == 201
        body = resp.json()["data"]
        pattern = (
            rf"{re.escape(CHATBOT_REF_PREFIX)}[0-9a-f]{{{CHATBOT_REF_SUFFIX_LENGTH}}}"
        )
        assert re.fullmatch(pattern, body["chatbot_ref"])
        [stored] = _sessions_for_contact(db, contact_ref)
        assert str(stored.id) == body["session_id"]
        assert stored.chatbot_ref == body["chatbot_ref"]

    def test_provided_chatbot_ref_is_used_as_is(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        chatbot_ref = f"Glific Flow {random_lower_string()}"

        resp = self._create(
            client, headers, anthropic_config, contact_ref, chatbot_ref=chatbot_ref
        )

        assert resp.status_code == 201
        assert resp.json()["data"]["chatbot_ref"] == chatbot_ref
        [stored] = _sessions_for_contact(db, contact_ref)
        assert stored.chatbot_ref == chatbot_ref

    def test_reusing_returned_ref_supersedes_previous_session(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        first = self._create(client, headers, anthropic_config, contact_ref).json()[
            "data"
        ]

        resp = self._create(
            client,
            headers,
            anthropic_config,
            contact_ref,
            chatbot_ref=first["chatbot_ref"],
        )

        assert resp.status_code == 201
        second = resp.json()["data"]
        assert second["chatbot_ref"] == first["chatbot_ref"]
        assert second["session_id"] != first["session_id"]
        latest = client.get(
            SESSIONS_URL,
            headers=headers,
            params={"contact_ref": contact_ref, "chatbot_ref": first["chatbot_ref"]},
        ).json()["data"]
        assert latest["session_id"] == second["session_id"]
        assert latest["status"] == "active"
        assert latest["completion_status"] == "in_progress"
        statuses = {
            str(s.id): (s.status, s.completion_status)
            for s in _sessions_for_contact(db, contact_ref)
        }
        assert statuses == {
            first["session_id"]: (
                ChatbotSessionStatusEnum.INACTIVE,
                ChatbotCompletionStatusEnum.ABANDONED,
            ),
            second["session_id"]: (
                ChatbotSessionStatusEnum.ACTIVE,
                ChatbotCompletionStatusEnum.IN_PROGRESS,
            ),
        }

    @pytest.mark.parametrize("chatbot_ref", ["", "x" * 256], ids=["empty", "too-long"])
    def test_invalid_chatbot_ref_returns_422(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
        chatbot_ref: str,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"

        resp = self._create(
            client, headers, anthropic_config, contact_ref, chatbot_ref=chatbot_ref
        )

        assert resp.status_code == 422
        assert _sessions_for_contact(db, contact_ref) == []

    def test_ref_generation_exhausted_returns_503_and_creates_nothing(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        create_test_chatbot_session(
            db, config=anthropic_config, chatbot_ref="LLMResponseWithPrompt-abcde"
        )
        # Randomness is the boundary: every draw collides with the ref above.
        monkeypatch.setattr(
            chatbot_ref_module.secrets, "randbelow", lambda _bound: 0xABCDE
        )
        contact_ref = f"contact-{random_lower_string()}"

        resp = self._create(client, headers, anthropic_config, contact_ref)

        assert resp.status_code == 503
        assert "ChatbotRefGenerationError" in resp.json()["error"]
        assert _sessions_for_contact(db, contact_ref) == []

    def test_concurrent_create_conflict_returns_409(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The real unique-index race is covered in the CRUD tests; there the
        # rollback would also discard this fixture's transaction mid-request.
        def lose_race(*_args: Any, **_kwargs: Any) -> ChatbotSession:
            raise ChatbotSessionConflictError(CONCURRENT_SESSION_CONFLICT_MESSAGE)

        monkeypatch.setattr(session_routes, "create_chatbot_session", lose_race)

        resp = self._create(
            client,
            headers,
            anthropic_config,
            f"contact-{random_lower_string()}",
            chatbot_ref=f"chatbot-{random_lower_string()}",
        )

        assert resp.status_code == 409
        assert resp.json()["error"] == CONCURRENT_SESSION_CONFLICT_MESSAGE


class TestGetLatestChatbotSession:
    def test_returns_contacts_newest_session_in_callers_project(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        chatbot_ref = f"chatbot-{random_lower_string()}"
        create_test_chatbot_session(
            db,
            config=anthropic_config,
            contact_ref=contact_ref,
            chatbot_ref=chatbot_ref,
        )
        newer = create_test_chatbot_session(
            db,
            config=anthropic_config,
            name="Onboarding Bot",
            contact_ref=contact_ref,
            chatbot_ref=chatbot_ref,
            config_version=1,
            system_prompt="Collect the student's name and grade.",
        )
        create_test_chatbot_session(
            db, config=anthropic_config, chatbot_ref=chatbot_ref
        )
        foreign_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        create_test_chatbot_session(
            db, config=foreign_config, contact_ref=contact_ref, chatbot_ref=chatbot_ref
        )

        resp = client.get(
            SESSIONS_URL,
            headers=headers,
            params={"contact_ref": contact_ref, "chatbot_ref": chatbot_ref},
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(newer.id)
        assert body["name"] == "Onboarding Bot"
        assert body["contact_ref"] == contact_ref
        assert body["chatbot_ref"] == chatbot_ref
        assert body["system_prompt"] == "Collect the student's name and grade."
        assert body["status"] == "active"
        assert body["completion_status"] == "in_progress"
        assert body["collected_fields"] == {}
        assert body["config_id"] == str(anthropic_config.id)
        assert body["config_version"] == 1

    def test_reports_stored_field_progress(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot_session: ChatbotSession,
    ) -> None:
        update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha", "city": "Pune"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )

        resp = client.get(
            SESSIONS_URL,
            headers=headers,
            params={
                "contact_ref": chatbot_session.contact_ref,
                "chatbot_ref": chatbot_session.chatbot_ref,
            },
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(chatbot_session.id)
        assert body["completion_status"] == "complete"
        assert body["collected_fields"] == {"name": "Asha", "city": "Pune"}

    def test_unknown_contact_returns_404(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        resp = client.get(
            SESSIONS_URL,
            headers=headers,
            params={
                "contact_ref": f"contact-{random_lower_string()}",
                "chatbot_ref": f"chatbot-{random_lower_string()}",
            },
        )

        assert resp.status_code == 404

    def test_missing_contact_ref_returns_422(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        resp = client.get(SESSIONS_URL, headers=headers)

        assert resp.status_code == 422

    def test_omitted_chatbot_ref_returns_contacts_newest_session_across_chatbots(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        create_test_chatbot_session(
            db,
            config=anthropic_config,
            contact_ref=contact_ref,
            chatbot_ref=f"chatbot-{random_lower_string()}",
        )
        newest = create_test_chatbot_session(
            db,
            config=anthropic_config,
            contact_ref=contact_ref,
            chatbot_ref=f"chatbot-{random_lower_string()}",
        )

        resp = client.get(
            SESSIONS_URL, headers=headers, params={"contact_ref": contact_ref}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(newest.id)
        assert body["chatbot_ref"] == newest.chatbot_ref

    def test_omitted_chatbot_ref_unknown_contact_returns_404(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        resp = client.get(
            SESSIONS_URL,
            headers=headers,
            params={"contact_ref": f"contact-{random_lower_string()}"},
        )

        assert resp.status_code == 404
        assert resp.json()["error"] == "No chatbot session found for this contact_ref"

    def test_unknown_chatbot_ref_for_known_contact_returns_404(
        self,
        client: TestClient,
        headers: dict[str, str],
        chatbot_session: ChatbotSession,
    ) -> None:
        resp = client.get(
            SESSIONS_URL,
            headers=headers,
            params={
                "contact_ref": chatbot_session.contact_ref,
                "chatbot_ref": f"chatbot-{random_lower_string()}",
            },
        )

        assert resp.status_code == 404


class TestSendChatbotSessionMessage:
    def test_stores_user_message_and_queues_turn(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        user_api_key: TestAuthContext,
        chatbot_session: ChatbotSession,
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = client.post(
            _message_url(chatbot_session.id),
            headers=headers,
            json={"message": "Asha", "webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 202
        assert resp.json()["data"] == {"status": "queued"}
        stored = list_chatbot_session_messages(db, session_id=chatbot_session.id)
        assert [(m.role, m.content) for m in stored] == [
            (ChatbotMessageRoleEnum.USER, "Asha")
        ]
        [job] = enqueued
        assert job["session_id"] == str(chatbot_session.id)
        assert job["organization_id"] == user_api_key.organization_id
        assert job["project_id"] == user_api_key.project_id
        assert job["webhook_url"] == WEBHOOK_URL

    def test_omitted_message_on_first_call_queues_greeting(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot_session: ChatbotSession,
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = client.post(
            _message_url(chatbot_session.id),
            headers=headers,
            json={"webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 202
        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []
        assert [job["session_id"] for job in enqueued] == [str(chatbot_session.id)]

    def test_omitted_message_after_session_started_returns_422(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot_session: ChatbotSession,
        enqueued: list[dict[str, Any]],
    ) -> None:
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [(ChatbotMessageRoleEnum.ASSISTANT, "Hi! What's your name?")],
        )

        resp = client.post(
            _message_url(chatbot_session.id),
            headers=headers,
            json={"webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 422
        assert "`message` is required" in resp.json()["error"]
        assert enqueued == []

    def test_unknown_session_returns_404(
        self,
        client: TestClient,
        headers: dict[str, str],
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = client.post(
            _message_url(uuid4()),
            headers=headers,
            json={"message": "Asha", "webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 404
        assert enqueued == []

    def test_session_in_other_project_returns_404(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        enqueued: list[dict[str, Any]],
    ) -> None:
        foreign_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        foreign_session = create_test_chatbot_session(db, config=foreign_config)

        resp = client.post(
            _message_url(foreign_session.id),
            headers=headers,
            json={"message": "Asha", "webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 404
        assert list_chatbot_session_messages(db, session_id=foreign_session.id) == []
        assert enqueued == []

    def test_non_https_webhook_url_returns_422(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot_session: ChatbotSession,
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = client.post(
            _message_url(chatbot_session.id),
            headers=headers,
            json={"message": "Asha", "webhook_url": "http://hooks.example.com/chatbot"},
        )

        assert resp.status_code == 422
        assert "Invalid webhook_url" in resp.json()["error"]
        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []
        assert enqueued == []

    @pytest.mark.parametrize(
        "payload",
        [{"message": "Asha"}, {}],
        ids=["with-message", "message-omitted"],
    )
    def test_inactive_session_returns_409_and_stores_nothing(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot_session: ChatbotSession,
        enqueued: list[dict[str, Any]],
        payload: dict[str, str],
    ) -> None:
        _supersede(db, chatbot_session)

        resp = client.post(
            _message_url(chatbot_session.id),
            headers=headers,
            json={**payload, "webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 409
        assert (
            resp.json()["error"] == "Chatbot session is inactive; start a new session"
        )
        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []
        assert enqueued == []

    def test_inactive_session_in_other_project_returns_404(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        enqueued: list[dict[str, Any]],
    ) -> None:
        foreign_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        foreign_session = create_test_chatbot_session(db, config=foreign_config)
        _supersede(db, foreign_session)

        resp = client.post(
            _message_url(foreign_session.id),
            headers=headers,
            json={"message": "Asha", "webhook_url": WEBHOOK_URL},
        )

        assert resp.status_code == 404
        assert enqueued == []
