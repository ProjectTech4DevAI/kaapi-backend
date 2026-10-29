from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import (
    Chatbot,
    ChatbotCompletionStatusEnum,
    ChatbotSession,
    ChatbotSessionStatusEnum,
    Project,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import (
    CHATBOTS_URL,
    LATEST_SESSION_URL,
    UNKNOWN_CHATBOT_ID,
    chatbot_url,
    create_test_chatbot,
    create_test_chatbot_session,
)


class TestCreateChatbot:
    def test_creates_chatbot_in_callers_project(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        user_api_key: TestAuthContext,
    ) -> None:
        resp = client.post(CHATBOTS_URL, headers=headers, json={"name": "Onboarding"})

        assert resp.status_code == 201
        body = resp.json()["data"]
        assert body["name"] == "Onboarding"
        assert body["project_id"] == user_api_key.project_id
        assert body["organization_id"] == user_api_key.organization_id
        stored = db.get(Chatbot, UUID(body["id"]))
        assert stored is not None
        assert stored.name == "Onboarding"
        assert stored.project_id == user_api_key.project_id

    @pytest.mark.parametrize(
        "payload",
        [{}, {"name": ""}, {"name": "x" * 256}],
        ids=["missing", "empty", "too-long"],
    )
    def test_invalid_name_returns_422(
        self, client: TestClient, headers: dict[str, str], payload: dict[str, str]
    ) -> None:
        resp = client.post(CHATBOTS_URL, headers=headers, json=payload)

        assert resp.status_code == 422


class TestListChatbots:
    def test_lists_only_callers_non_deleted_chatbots(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        project: Project,
        chatbot: Chatbot,
        foreign_chatbot: Chatbot,
    ) -> None:
        deleted = create_test_chatbot(db, project=project)
        assert (
            client.delete(chatbot_url(deleted.id), headers=headers).status_code == 200
        )

        resp = client.get(CHATBOTS_URL, headers=headers)

        assert resp.status_code == 200
        ids = {item["id"] for item in resp.json()["data"]}
        assert str(chatbot.id) in ids
        assert str(foreign_chatbot.id) not in ids
        assert str(deleted.id) not in ids

    def test_skip_and_limit_page_newest_first(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        project: Project,
    ) -> None:
        # The seeded project may already own chatbots, so page through a known head.
        create_test_chatbot(db, project=project)
        middle = create_test_chatbot(db, project=project)
        newest = create_test_chatbot(db, project=project)

        first = client.get(CHATBOTS_URL, headers=headers, params={"limit": 1})
        second = client.get(
            CHATBOTS_URL, headers=headers, params={"skip": 1, "limit": 1}
        )

        assert [c["id"] for c in first.json()["data"]] == [str(newest.id)]
        assert [c["id"] for c in second.json()["data"]] == [str(middle.id)]


class TestGetChatbot:
    def test_returns_chatbot(
        self, client: TestClient, headers: dict[str, str], chatbot: Chatbot
    ) -> None:
        resp = client.get(chatbot_url(chatbot.id), headers=headers)

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["id"] == str(chatbot.id)
        assert body["name"] == chatbot.name

    def test_unknown_returns_404(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        resp = client.get(chatbot_url(UNKNOWN_CHATBOT_ID), headers=headers)

        assert resp.status_code == 404
        assert resp.json()["error"] == "Chatbot not found"

    def test_other_projects_chatbot_returns_404(
        self, client: TestClient, headers: dict[str, str], foreign_chatbot: Chatbot
    ) -> None:
        resp = client.get(chatbot_url(foreign_chatbot.id), headers=headers)

        assert resp.status_code == 404


class TestUpdateChatbot:
    def test_renames_chatbot(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot: Chatbot,
    ) -> None:
        resp = client.patch(
            chatbot_url(chatbot.id), headers=headers, json={"name": "Renamed"}
        )

        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Renamed"
        db.expire_all()
        stored = db.get(Chatbot, chatbot.id)
        assert stored is not None
        assert stored.name == "Renamed"

    def test_other_projects_chatbot_returns_404_and_is_unchanged(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        foreign_chatbot: Chatbot,
    ) -> None:
        original_name = foreign_chatbot.name

        resp = client.patch(
            chatbot_url(foreign_chatbot.id), headers=headers, json={"name": "Hijack"}
        )

        assert resp.status_code == 404
        db.expire_all()
        stored = db.get(Chatbot, foreign_chatbot.id)
        assert stored is not None
        assert stored.name == original_name


class TestDeleteChatbot:
    def test_soft_deletes_and_hides_chatbot(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        chatbot: Chatbot,
    ) -> None:
        resp = client.delete(chatbot_url(chatbot.id), headers=headers)

        assert resp.status_code == 200
        assert resp.json()["data"] == {"message": "Chatbot deleted successfully"}
        db.expire_all()
        stored = db.get(Chatbot, chatbot.id)
        assert stored is not None
        assert stored.deleted_at is not None
        assert client.get(chatbot_url(chatbot.id), headers=headers).status_code == 404
        assert (
            client.patch(
                chatbot_url(chatbot.id), headers=headers, json={"name": "x"}
            ).status_code
            == 404
        )
        assert (
            client.delete(chatbot_url(chatbot.id), headers=headers).status_code == 404
        )

    def test_deactivates_active_sessions_and_hides_them_from_latest(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        live_chatbot: Chatbot,
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=live_chatbot)

        resp = client.delete(chatbot_url(live_chatbot.id), headers=headers)

        assert resp.status_code == 200
        db.expire_all()
        stored = db.get(ChatbotSession, chatbot_session.id)
        assert stored is not None
        assert stored.status == ChatbotSessionStatusEnum.INACTIVE
        assert stored.completion_status == ChatbotCompletionStatusEnum.ABANDONED
        latest = client.get(
            LATEST_SESSION_URL,
            headers=headers,
            params={"contact_ref": chatbot_session.contact_ref},
        )
        assert latest.status_code == 404

    def test_other_projects_chatbot_returns_404_and_is_kept(
        self,
        client: TestClient,
        db: Session,
        headers: dict[str, str],
        foreign_chatbot: Chatbot,
    ) -> None:
        resp = client.delete(chatbot_url(foreign_chatbot.id), headers=headers)

        assert resp.status_code == 404
        db.expire_all()
        stored = db.get(Chatbot, foreign_chatbot.id)
        assert stored is not None
        assert stored.deleted_at is None
