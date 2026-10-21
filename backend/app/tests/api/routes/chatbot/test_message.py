import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Config
from app.services.chatbot import engine
from app.tests.services.chatbot.test_engine import ScriptedChatModel
from app.tests.utils.auth import TestAuthContext
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


def _patch_chat_model(
    monkeypatch: pytest.MonkeyPatch, chat_model: ScriptedChatModel
) -> None:
    # The Anthropic SDK is the external boundary; config resolution stays on the real DB.
    monkeypatch.setattr(engine, "get_chat_model", lambda _settings: chat_model)


class TestSendChatbotMessage:
    def test_returns_updated_transcript(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        anthropic_config: Config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_chat_model(monkeypatch, ScriptedChatModel(replies=["Nice, which city?"]))

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
