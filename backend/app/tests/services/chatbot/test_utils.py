import pytest
from langchain_anthropic import ChatAnthropic

from app.core.config import settings
from app.services.chatbot.utils import (
    ChatbotModelSettings,
    default_thinking,
    get_chat_model,
)

PROJECT_API_KEY = "sk-ant-project"
PLATFORM_API_KEY = "sk-ant-platform"


class TestChatbotModelSettings:
    def test_defaults(self) -> None:
        model_settings = ChatbotModelSettings(model="claude-sonnet-4-5")

        assert model_settings.max_tokens == 1024
        assert model_settings.reasoning_effort is None
        assert model_settings.thinking == {"type": "adaptive"}
        assert model_settings.history_token_limit == 4000

    def test_thinking_default_is_not_shared_between_instances(self) -> None:
        first = ChatbotModelSettings(model="claude-sonnet-4-5")
        first.thinking["budget_tokens"] = 2048

        second = ChatbotModelSettings(model="claude-sonnet-4-5")

        assert second.thinking == {"type": "adaptive"}
        assert default_thinking() == {"type": "adaptive"}

    def test_repr_hides_the_api_key(self) -> None:
        model_settings = ChatbotModelSettings(
            model="claude-sonnet-4-5", api_key=PROJECT_API_KEY
        )

        assert PROJECT_API_KEY not in repr(model_settings)
        assert "claude-sonnet-4-5" in repr(model_settings)


class TestGetChatModel:
    @pytest.fixture(autouse=True)
    def platform_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A platform/env key must never be used in place of the project's key.
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", PLATFORM_API_KEY)
        monkeypatch.setenv("ANTHROPIC_API_KEY", PLATFORM_API_KEY)

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(RuntimeError, match="model_settings has no api_key"):
            get_chat_model(ChatbotModelSettings(model="claude-sonnet-4-5"))

    def test_builds_model_from_settings(self) -> None:
        chat_model = get_chat_model(
            ChatbotModelSettings(
                model="claude-sonnet-4-5",
                max_tokens=512,
                reasoning_effort="high",
                thinking={"type": "enabled", "budget_tokens": 2048},
                api_key=PROJECT_API_KEY,
            )
        )

        assert isinstance(chat_model, ChatAnthropic)
        assert chat_model.model == "claude-sonnet-4-5"
        assert chat_model.max_tokens == 512
        assert chat_model.reasoning_effort == "high"
        assert chat_model.thinking == {"type": "enabled", "budget_tokens": 2048}
        assert chat_model.anthropic_api_key.get_secret_value() == PROJECT_API_KEY

    def test_omits_reasoning_effort_when_unset(self) -> None:
        chat_model = get_chat_model(
            ChatbotModelSettings(model="claude-sonnet-4-5", api_key=PROJECT_API_KEY)
        )

        assert chat_model.reasoning_effort is None
        assert chat_model.thinking == {"type": "adaptive"}
