import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Config
from app.services.chatbot.agents.onboarding.types import FieldExtraction
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import ScriptedChatModel, patch_chat_model
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
)

MESSAGE_URL = f"{settings.API_V1_STR}/chatbot/message"
SYSTEM_PROMPT = "Collect the user's name and city."


@pytest.fixture
def anthropic_config(db: Session, user_api_key: TestAuthContext) -> Config:
    return create_test_config(
        db,
        project_id=user_api_key.project_id,
        config_blob=build_anthropic_text_config_blob(),
    )


class TestSendChatbotMessage:
    def test_returns_updated_transcript(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(monkeypatch, ScriptedChatModel(replies=["Nice, which city?"]))

        resp = client.post(
            MESSAGE_URL,
            headers={"X-API-KEY": user_api_key.key},
            json={
                "config_id": str(anthropic_config.id),
                "system_prompt": SYSTEM_PROMPT,
                "history": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "assistant", "content": "Hi! What's your name?"},
                ],
                "message": "Asha",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["data"]["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! What's your name?"},
            {"role": "user", "content": "Asha"},
            {"role": "assistant", "content": "Nice, which city?"},
        ]

    def test_returns_extracted_fields_and_completion_status(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(
                replies=["Thanks, Asha from Pune!"],
                extractions=[
                    FieldExtraction(
                        collected_fields={"name": "Asha", "city": "Pune"},
                        missing_fields=[],
                        is_complete=True,
                    )
                ],
            ),
        )

        resp = client.post(
            MESSAGE_URL,
            headers={"X-API-KEY": user_api_key.key},
            json={
                "config_id": str(anthropic_config.id),
                "system_prompt": SYSTEM_PROMPT,
                "history": [
                    {"role": "assistant", "content": "Hi! What's your name and city?"}
                ],
                "message": "Asha, Pune",
            },
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["collected_fields"] == {"name": "Asha", "city": "Pune"}
        assert body["completion_status"] == "complete"

    def test_greeting_returns_no_fields_in_progress(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_chat_model(
            monkeypatch, ScriptedChatModel(replies=["Hi! What's your name?"])
        )

        resp = client.post(
            MESSAGE_URL,
            headers={"X-API-KEY": user_api_key.key},
            json={
                "config_id": str(anthropic_config.id),
                "system_prompt": SYSTEM_PROMPT,
            },
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["collected_fields"] == {}
        assert body["completion_status"] == "in_progress"

    def test_missing_message_with_history_returns_422(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_model = ScriptedChatModel()
        patch_chat_model(monkeypatch, chat_model)

        resp = client.post(
            MESSAGE_URL,
            headers={"X-API-KEY": user_api_key.key},
            json={
                "config_id": str(anthropic_config.id),
                "system_prompt": SYSTEM_PROMPT,
                "history": [{"role": "assistant", "content": "Hi! What's your name?"}],
            },
        )

        assert resp.status_code == 422
        assert "`message` is required" in resp.json()["error"]
        assert chat_model.calls == []

    def test_provider_failure_returns_502(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        patch_chat_model(
            monkeypatch,
            ScriptedChatModel(error=anthropic.APIConnectionError(request=request)),
        )

        resp = client.post(
            MESSAGE_URL,
            headers={"X-API-KEY": user_api_key.key},
            json={
                "config_id": str(anthropic_config.id),
                "system_prompt": SYSTEM_PROMPT,
                "message": "Asha",
            },
        )

        assert resp.status_code == 502
        assert "(code: APIConnectionError)" in resp.json()["error"]
