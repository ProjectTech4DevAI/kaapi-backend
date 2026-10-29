import socket
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlmodel import Session

from app.api.routes.chatbot import session as session_routes
from app.crud.chatbot import (
    delete_chatbot,
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.models import (
    Chatbot,
    ChatbotCompletionStatusEnum,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionStatusEnum,
    Config,
)
from app.services.chatbot import session as session_service
from app.services.chatbot.exceptions import (
    CONCURRENT_SESSION_CONFLICT_MESSAGE,
    ChatbotSessionConflictError,
)
from app.services.chatbot.session import NO_LIVE_FLOW_MESSAGE
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import (
    DEFAULT_SYSTEM_PROMPT,
    LATEST_SESSION_URL,
    ONBOARDING_NODE_ID,
    UNKNOWN_CHATBOT_ID,
    WEBHOOK_URL,
    add_test_chatbot_messages,
    build_flow_edge,
    build_flow_node,
    build_onboarding_node,
    create_test_chatbot_session,
    create_test_flow_version,
    create_test_live_chatbot,
    list_stored_sessions_for_contact,
    message_url,
    random_contact_ref,
    sessions_url,
)

PUBLIC_IP = "93.184.216.34"


def _supersede(db: Session, chatbot_session: ChatbotSession) -> ChatbotSession:
    chatbot = db.get(Chatbot, chatbot_session.chatbot_id)
    assert chatbot is not None
    return create_test_chatbot_session(
        db, chatbot=chatbot, contact_ref=chatbot_session.contact_ref
    )


@pytest.fixture
def chatbot_session(db: Session, live_chatbot: Chatbot) -> ChatbotSession:
    return create_test_chatbot_session(db, chatbot=live_chatbot)


@pytest.fixture
def create_session(
    client: TestClient, user_api_key_header: dict[str, str]
) -> Callable[[UUID, dict[str, Any]], Response]:
    return lambda chatbot_id, payload: client.post(
        sessions_url(chatbot_id), headers=user_api_key_header, json=payload
    )


@pytest.fixture
def get_latest(
    client: TestClient, user_api_key_header: dict[str, str]
) -> Callable[..., Response]:
    def _get_latest(contact_ref: str, chatbot_id: UUID | None = None) -> Response:
        params = {"contact_ref": contact_ref}
        if chatbot_id is not None:
            params["chatbot_id"] = str(chatbot_id)
        return client.get(
            LATEST_SESSION_URL, headers=user_api_key_header, params=params
        )

    return _get_latest


@pytest.fixture
def send_message(
    client: TestClient, user_api_key_header: dict[str, str]
) -> Callable[..., Response]:
    def _send_message(session_id: UUID, **body: str) -> Response:
        return client.post(
            message_url(session_id),
            headers=user_api_key_header,
            json={"webhook_url": WEBHOOK_URL, **body},
        )

    return _send_message


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
    def test_creates_session_pinned_to_live_flow_and_entry_node(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
    ) -> None:
        live = create_test_flow_version(
            db,
            chatbot=chatbot,
            nodes=[
                build_flow_node("start", "input"),
                build_onboarding_node(config_id=config.id),
            ],
            edges=[build_flow_edge("e1", "start", ONBOARDING_NODE_ID)],
            is_live=True,
        )
        # A newer unpublished draft must not be picked up.
        create_test_flow_version(
            db, chatbot=chatbot, nodes=[build_onboarding_node(config_id=config.id)]
        )
        contact_ref = random_contact_ref()

        resp = create_session(chatbot.id, {"contact_ref": contact_ref})

        assert resp.status_code == 201
        body = resp.json()["data"]
        assert body["chatbot_id"] == str(chatbot.id)
        assert body["flow_version"] == 1
        assert body["active_node_id"] == ONBOARDING_NODE_ID
        [stored] = list_stored_sessions_for_contact(db, contact_ref)
        assert str(stored.id) == body["session_id"]
        assert stored.flow_version_id == live.id
        assert stored.active_node_id == ONBOARDING_NODE_ID
        assert stored.status == ChatbotSessionStatusEnum.ACTIVE
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS

    def test_new_session_for_same_contact_supersedes_previous(
        self,
        db: Session,
        live_chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
        get_latest: Callable[..., Response],
    ) -> None:
        contact_ref = random_contact_ref()
        payload = {"contact_ref": contact_ref}
        first = create_session(live_chatbot.id, payload)

        resp = create_session(live_chatbot.id, payload)

        assert resp.status_code == 201
        first_id = first.json()["data"]["session_id"]
        second_id = resp.json()["data"]["session_id"]
        assert second_id != first_id
        latest = get_latest(contact_ref, live_chatbot.id).json()["data"]
        assert latest["session_id"] == second_id
        statuses = {
            str(s.id): (s.status, s.completion_status)
            for s in list_stored_sessions_for_contact(db, contact_ref)
        }
        assert statuses == {
            first_id: (
                ChatbotSessionStatusEnum.INACTIVE,
                ChatbotCompletionStatusEnum.ABANDONED,
            ),
            second_id: (
                ChatbotSessionStatusEnum.ACTIVE,
                ChatbotCompletionStatusEnum.IN_PROGRESS,
            ),
        }

    def test_chatbot_without_live_flow_returns_409(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
    ) -> None:
        create_test_flow_version(
            db, chatbot=chatbot, nodes=[build_onboarding_node(config_id=config.id)]
        )
        contact_ref = random_contact_ref()

        resp = create_session(chatbot.id, {"contact_ref": contact_ref})

        assert resp.status_code == 409
        assert resp.json()["error"] == NO_LIVE_FLOW_MESSAGE
        assert list_stored_sessions_for_contact(db, contact_ref) == []

    def test_unknown_chatbot_returns_404(
        self, create_session: Callable[[UUID, dict[str, Any]], Response]
    ) -> None:
        resp = create_session(UNKNOWN_CHATBOT_ID, {"contact_ref": "contact-1"})

        assert resp.status_code == 404
        assert resp.json()["error"] == "Chatbot not found"

    def test_other_projects_chatbot_returns_404(
        self,
        db: Session,
        foreign_live_chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
    ) -> None:
        contact_ref = random_contact_ref()

        resp = create_session(foreign_live_chatbot.id, {"contact_ref": contact_ref})

        assert resp.status_code == 404
        assert list_stored_sessions_for_contact(db, contact_ref) == []

    def test_deleted_chatbot_returns_404(
        self,
        db: Session,
        live_chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
    ) -> None:
        delete_chatbot(db, chatbot=live_chatbot)

        resp = create_session(live_chatbot.id, {"contact_ref": "c-1"})

        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "payload",
        [{}, {"contact_ref": ""}, {"contact_ref": "x" * 501}],
        ids=["missing", "empty", "too-long"],
    )
    def test_invalid_contact_ref_returns_422(
        self,
        live_chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
        payload: dict[str, str],
    ) -> None:
        resp = create_session(live_chatbot.id, payload)

        assert resp.status_code == 422

    def test_concurrent_create_conflict_returns_409(
        self,
        live_chatbot: Chatbot,
        create_session: Callable[[UUID, dict[str, Any]], Response],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The real unique-index race is covered in the CRUD tests; there the
        # rollback would also discard this fixture's transaction mid-request.
        def lose_race(*_args: Any, **_kwargs: Any) -> ChatbotSession:
            raise ChatbotSessionConflictError(CONCURRENT_SESSION_CONFLICT_MESSAGE)

        monkeypatch.setattr(session_service, "create_chatbot_session", lose_race)

        resp = create_session(live_chatbot.id, {"contact_ref": random_contact_ref()})

        assert resp.status_code == 409
        assert resp.json()["error"] == CONCURRENT_SESSION_CONFLICT_MESSAGE


class TestGetLatestChatbotSession:
    def test_returns_contacts_newest_session_with_chatbot_and_node_details(
        self,
        db: Session,
        config: Config,
        foreign_live_chatbot: Chatbot,
        get_latest: Callable[..., Response],
    ) -> None:
        chatbot, _ = create_test_live_chatbot(
            db,
            config=config,
            name="Onboarding Bot",
            prompt="Collect the student's name and grade.",
        )
        contact_ref = random_contact_ref()
        create_test_chatbot_session(db, chatbot=chatbot, contact_ref=contact_ref)
        newer = create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=contact_ref
        )
        create_test_chatbot_session(db, chatbot=chatbot)
        create_test_chatbot_session(
            db, chatbot=foreign_live_chatbot, contact_ref=contact_ref
        )

        resp = get_latest(contact_ref, chatbot.id)

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(newer.id)
        assert body["chatbot_id"] == str(chatbot.id)
        assert body["chatbot_name"] == "Onboarding Bot"
        assert body["flow_version"] == 1
        assert body["active_node_id"] == ONBOARDING_NODE_ID
        assert body["prompt"] == "Collect the student's name and grade."
        assert body["contact_ref"] == contact_ref
        assert body["status"] == "active"
        assert body["completion_status"] == "in_progress"
        assert body["collected_fields"] == {}

    def test_reports_stored_field_progress(
        self,
        db: Session,
        chatbot_session: ChatbotSession,
        get_latest: Callable[..., Response],
    ) -> None:
        update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha", "city": "Pune"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )

        resp = get_latest(chatbot_session.contact_ref, chatbot_session.chatbot_id)

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(chatbot_session.id)
        assert body["completion_status"] == "complete"
        assert body["collected_fields"] == {"name": "Asha", "city": "Pune"}

    def test_prompt_comes_from_the_sessions_pinned_flow(
        self,
        db: Session,
        config: Config,
        live_chatbot: Chatbot,
        chatbot_session: ChatbotSession,
        get_latest: Callable[..., Response],
    ) -> None:
        create_test_flow_version(
            db,
            chatbot=live_chatbot,
            nodes=[
                build_onboarding_node(config_id=config.id, prompt="A newer prompt.")
            ],
            is_live=True,
        )

        resp = get_latest(chatbot_session.contact_ref)

        body = resp.json()["data"]
        assert body["flow_version"] == 1
        assert body["prompt"] == DEFAULT_SYSTEM_PROMPT

    def test_prompt_is_null_when_active_node_is_not_in_flow(
        self,
        db: Session,
        live_chatbot: Chatbot,
        get_latest: Callable[..., Response],
    ) -> None:
        chatbot_session = create_test_chatbot_session(
            db, chatbot=live_chatbot, active_node_id="ghost"
        )

        resp = get_latest(chatbot_session.contact_ref)

        assert resp.status_code == 200
        assert resp.json()["data"]["active_node_id"] == "ghost"
        assert resp.json()["data"]["prompt"] is None

    def test_omitted_chatbot_id_returns_newest_session_across_chatbots(
        self,
        db: Session,
        config: Config,
        chatbot_session: ChatbotSession,
        get_latest: Callable[..., Response],
    ) -> None:
        other_chatbot, _ = create_test_live_chatbot(db, config=config)
        newest = create_test_chatbot_session(
            db, chatbot=other_chatbot, contact_ref=chatbot_session.contact_ref
        )

        resp = get_latest(chatbot_session.contact_ref)

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == str(newest.id)
        assert body["chatbot_id"] == str(other_chatbot.id)

    def test_deleted_chatbots_sessions_are_excluded(
        self,
        db: Session,
        config: Config,
        chatbot_session: ChatbotSession,
        get_latest: Callable[..., Response],
    ) -> None:
        deleted_chatbot, _ = create_test_live_chatbot(db, config=config)
        create_test_chatbot_session(
            db, chatbot=deleted_chatbot, contact_ref=chatbot_session.contact_ref
        )
        delete_chatbot(db, chatbot=deleted_chatbot)

        across = get_latest(chatbot_session.contact_ref)
        scoped = get_latest(chatbot_session.contact_ref, deleted_chatbot.id)

        assert across.json()["data"]["session_id"] == str(chatbot_session.id)
        assert scoped.status_code == 404

    def test_unknown_contact_returns_404(
        self, live_chatbot: Chatbot, get_latest: Callable[..., Response]
    ) -> None:
        resp = get_latest(random_contact_ref(), live_chatbot.id)

        assert resp.status_code == 404
        assert (
            resp.json()["error"]
            == "No chatbot session found for this contact_ref and chatbot_id"
        )

    def test_omitted_chatbot_id_unknown_contact_returns_404(
        self, get_latest: Callable[..., Response]
    ) -> None:
        resp = get_latest(random_contact_ref())

        assert resp.status_code == 404
        assert resp.json()["error"] == "No chatbot session found for this contact_ref"

    def test_other_projects_session_returns_404(
        self,
        db: Session,
        foreign_live_chatbot: Chatbot,
        get_latest: Callable[..., Response],
    ) -> None:
        foreign_session = create_test_chatbot_session(db, chatbot=foreign_live_chatbot)

        resp = get_latest(foreign_session.contact_ref)

        assert resp.status_code == 404

    def test_missing_contact_ref_returns_422(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        resp = client.get(LATEST_SESSION_URL, headers=user_api_key_header)

        assert resp.status_code == 422


class TestSendChatbotSessionMessage:
    def test_stores_user_message_and_queues_turn(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        chatbot_session: ChatbotSession,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = send_message(chatbot_session.id, message="Asha")

        assert resp.status_code == 202
        assert resp.json()["data"] == {"status": "queued"}
        stored = list_chatbot_session_messages(db, session_id=chatbot_session.id)
        assert [(m.role, m.content, m.node_id) for m in stored] == [
            (ChatbotMessageRoleEnum.USER, "Asha", ONBOARDING_NODE_ID)
        ]
        [job] = enqueued
        assert job["session_id"] == str(chatbot_session.id)
        assert job["organization_id"] == user_api_key.organization_id
        assert job["project_id"] == user_api_key.project_id
        assert job["webhook_url"] == WEBHOOK_URL

    def test_omitted_message_on_first_call_queues_greeting(
        self,
        db: Session,
        chatbot_session: ChatbotSession,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = send_message(chatbot_session.id)

        assert resp.status_code == 202
        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []
        assert [job["session_id"] for job in enqueued] == [str(chatbot_session.id)]

    def test_omitted_message_after_session_started_returns_422(
        self,
        db: Session,
        chatbot_session: ChatbotSession,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [(ChatbotMessageRoleEnum.ASSISTANT, "Hi! What's your name?")],
        )

        resp = send_message(chatbot_session.id)

        assert resp.status_code == 422
        assert "`message` is required" in resp.json()["error"]
        assert enqueued == []

    def test_unknown_session_returns_404(
        self,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = send_message(uuid4(), message="Asha")

        assert resp.status_code == 404
        assert enqueued == []

    def test_session_in_other_project_returns_404(
        self,
        db: Session,
        foreign_live_chatbot: Chatbot,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        foreign_session = create_test_chatbot_session(db, chatbot=foreign_live_chatbot)

        resp = send_message(foreign_session.id, message="Asha")

        assert resp.status_code == 404
        assert list_chatbot_session_messages(db, session_id=foreign_session.id) == []
        assert enqueued == []

    def test_non_https_webhook_url_returns_422(
        self,
        db: Session,
        chatbot_session: ChatbotSession,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        resp = send_message(
            chatbot_session.id,
            message="Asha",
            webhook_url="http://hooks.example.com/chatbot",
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
        db: Session,
        chatbot_session: ChatbotSession,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
        payload: dict[str, str],
    ) -> None:
        _supersede(db, chatbot_session)

        resp = send_message(chatbot_session.id, **payload)

        assert resp.status_code == 409
        assert (
            resp.json()["error"] == "Chatbot session is inactive; start a new session"
        )
        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []
        assert enqueued == []

    def test_inactive_session_in_other_project_returns_404(
        self,
        db: Session,
        foreign_live_chatbot: Chatbot,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        foreign_session = create_test_chatbot_session(db, chatbot=foreign_live_chatbot)
        _supersede(db, foreign_session)

        resp = send_message(foreign_session.id, message="Asha")

        assert resp.status_code == 404
        assert enqueued == []

    def test_session_of_deleted_chatbot_returns_409(
        self,
        db: Session,
        live_chatbot: Chatbot,
        chatbot_session: ChatbotSession,
        send_message: Callable[..., Response],
        enqueued: list[dict[str, Any]],
    ) -> None:
        delete_chatbot(db, chatbot=live_chatbot)

        resp = send_message(chatbot_session.id, message="Asha")

        assert resp.status_code == 409
        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []
        assert enqueued == []
