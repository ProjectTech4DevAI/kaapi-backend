import socket
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

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
    Config,
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
                "system_prompt": SYSTEM_PROMPT,
            },
        )

        assert resp.status_code == 422
        assert "only supports configs" in resp.json()["error"]


class TestGetLatestChatbotSession:
    def test_returns_contacts_newest_session_in_callers_project(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        anthropic_config: Config,
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        create_test_chatbot_session(
            db, config=anthropic_config, contact_ref=contact_ref
        )
        newer = create_test_chatbot_session(
            db,
            config=anthropic_config,
            name="Onboarding Bot",
            contact_ref=contact_ref,
            config_version=1,
        )
        create_test_chatbot_session(db, config=anthropic_config)
        foreign_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        create_test_chatbot_session(db, config=foreign_config, contact_ref=contact_ref)

        resp = client.get(
            SESSIONS_URL, headers=headers, params={"contact_ref": contact_ref}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(newer.id)
        assert body["name"] == "Onboarding Bot"
        assert body["contact_ref"] == contact_ref
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
            params={"contact_ref": chatbot_session.contact_ref},
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
            params={"contact_ref": f"contact-{random_lower_string()}"},
        )

        assert resp.status_code == 404

    def test_missing_contact_ref_returns_422(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        resp = client.get(SESSIONS_URL, headers=headers)

        assert resp.status_code == 422


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
