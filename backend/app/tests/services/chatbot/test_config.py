from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud.config import ConfigVersionCrud
from app.models import Config
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.exceptions import ChatbotInputError
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.chatbot import (
    add_test_config_version,
    create_test_anthropic_config,
)
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
    create_test_project,
    create_test_version,
)


def _resolve(
    db: Session,
    config: Config,
    *,
    config_version: int | None = None,
    max_tokens: int | None = None,
    history_token_limit: int | None = None,
) -> ChatbotModelSettings:
    return resolve_chatbot_model_settings(
        session=db,
        project_id=config.project_id,
        config_id=config.id,
        config_version=config_version,
        max_tokens=max_tokens,
        history_token_limit=history_token_limit,
    )


def _overwrite_latest_blob(db: Session, config: Config, blob: dict[str, Any]) -> None:
    # Bypasses ConfigVersionCrud validation to simulate a legacy/corrupt stored row.
    version = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    ).read_latest()
    assert version is not None
    version.config_blob = blob
    db.add(version)
    db.commit()


class TestResolveChatbotModelSettings:
    def test_latest_version_with_server_defaults(self, db: Session) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(model="claude-sonnet-4-6")
        )
        create_test_version(
            db,
            config_id=config.id,
            project_id=config.project_id,
            config_blob=build_anthropic_text_config_blob(
                model="claude-opus-5-5",
                effort="high",
                thinking={"type": "adaptive", "display": "summarized"},
            ),
        )

        model_settings = _resolve(db, config)

        assert model_settings == ChatbotModelSettings(
            model="claude-opus-5-5",
            max_tokens=1024,
            reasoning_effort="high",
            thinking={"type": "adaptive", "display": "summarized"},
            history_token_limit=4000,
        )

    def test_enabled_thinking_with_budget_below_max_tokens_is_accepted(
        self, db: Session
    ) -> None:
        thinking = {"type": "enabled", "budget_tokens": 2048}
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(thinking=thinking)
        )

        model_settings = _resolve(db, config, max_tokens=4096)

        assert model_settings.thinking == thinking
        assert model_settings.max_tokens == 4096

    @pytest.mark.parametrize(
        "thinking",
        [
            {"type": "enabled", "budget_tokens": 2048},
            {"type": "enabled", "budget_tokens": 1024},
            {"type": "enabled"},
        ],
        ids=["budget-above-max", "budget-equal-max", "budget-missing"],
    )
    def test_enabled_thinking_budget_not_below_max_tokens_rejected(
        self, db: Session, thinking: dict[str, Any]
    ) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(thinking=thinking)
        )

        with pytest.raises(ChatbotInputError, match="budget_tokens"):
            _resolve(db, config)

    def test_explicit_version_and_request_overrides(self, db: Session) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(model="claude-sonnet-4-6")
        )
        add_test_config_version(db, config)

        model_settings = _resolve(
            db, config, config_version=1, max_tokens=256, history_token_limit=900
        )

        assert model_settings.model == "claude-sonnet-4-6"
        assert model_settings.max_tokens == 256
        assert model_settings.history_token_limit == 900
        assert model_settings.reasoning_effort is None
        assert model_settings.thinking == {"type": "adaptive"}

    def test_unknown_version_raises_404(self, db: Session) -> None:
        config = create_test_anthropic_config(db)

        with pytest.raises(HTTPException) as exc_info:
            _resolve(db, config, config_version=99)

        assert exc_info.value.status_code == 404

    def test_config_from_other_project_raises_404(self, db: Session) -> None:
        config = create_test_anthropic_config(db)
        other_project = create_test_project(db)

        with pytest.raises(HTTPException) as exc_info:
            resolve_chatbot_model_settings(
                session=db,
                project_id=other_project.id,
                config_id=config.id,
                config_version=None,
                max_tokens=None,
                history_token_limit=None,
            )

        assert exc_info.value.status_code == 404

    def test_unknown_config_raises_404(self, db: Session) -> None:
        project = create_test_project(db)

        with pytest.raises(HTTPException) as exc_info:
            resolve_chatbot_model_settings(
                session=db,
                project_id=project.id,
                config_id=uuid4(),
                config_version=None,
                max_tokens=None,
                history_token_limit=None,
            )

        assert exc_info.value.status_code == 404

    def test_no_live_versions_raises_404(self, db: Session) -> None:
        config = create_test_anthropic_config(db)
        ConfigVersionCrud(
            session=db, project_id=config.project_id, config_id=config.id
        ).delete_or_raise(1)

        with pytest.raises(HTTPException) as exc_info:
            _resolve(db, config)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == f"No version found for config '{config.id}'"

    def test_non_anthropic_provider_rejected(self, db: Session) -> None:
        config = create_test_config(db, use_kaapi_schema=True)

        with pytest.raises(ChatbotInputError, match="only supports configs"):
            _resolve(db, config)

    def test_native_completion_config_rejected(self, db: Session) -> None:
        config = create_test_config(db)

        with pytest.raises(ChatbotInputError, match="only supports configs"):
            _resolve(db, config)

    def test_invalid_stored_blob_rejected(self, db: Session) -> None:
        config = create_test_anthropic_config(db)
        _overwrite_latest_blob(db, config, {"completion": {"provider": "anthropic"}})

        with pytest.raises(ChatbotInputError, match="blob is invalid"):
            _resolve(db, config)

    def test_missing_model_rejected(self, db: Session) -> None:
        config = create_test_anthropic_config(db)
        _overwrite_latest_blob(
            db,
            config,
            {"completion": {"provider": "anthropic", "type": "text", "params": {}}},
        )

        with pytest.raises(ChatbotInputError, match="missing completion.params.model"):
            _resolve(db, config)

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_non_anthropic_effort_rejected(self, db: Session, effort: str) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(effort=effort)
        )

        with pytest.raises(ChatbotInputError, match=f"'{effort}' is not an Anthropic"):
            _resolve(db, config)
