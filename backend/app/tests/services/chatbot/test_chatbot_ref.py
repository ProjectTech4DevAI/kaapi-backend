import logging
import re

import pytest
from sqlmodel import Session

from app.models import Config
from app.services.chatbot import chatbot_ref as chatbot_ref_module
from app.services.chatbot.chatbot_ref import (
    CHATBOT_REF_MAX_ATTEMPTS,
    CHATBOT_REF_PREFIX,
    CHATBOT_REF_SUFFIX_LENGTH,
    generate_chatbot_ref,
)
from app.services.chatbot.exceptions import ChatbotRefGenerationError
from app.tests.utils.chatbot import create_test_chatbot_session
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
)

USED_SUFFIX = 0xABCDE
USED_REF = "LLMResponseWithPrompt-abcde"
FRESH_SUFFIX = 0x1
FRESH_REF = "LLMResponseWithPrompt-00001"


@pytest.fixture
def config(db: Session) -> Config:
    return create_test_config(db, config_blob=build_anthropic_text_config_blob())


def _script_randbelow(monkeypatch: pytest.MonkeyPatch, values: list[int]) -> list[int]:
    """Replace the CSPRNG with a script; returns the list of bounds it was asked for."""
    bounds: list[int] = []
    remaining = iter(values)

    def fake_randbelow(bound: int) -> int:
        bounds.append(bound)
        return next(remaining)

    monkeypatch.setattr(chatbot_ref_module.secrets, "randbelow", fake_randbelow)
    return bounds


class TestGenerateChatbotRef:
    def test_ref_is_prefix_plus_lowercase_hex_suffix(
        self, db: Session, config: Config
    ) -> None:
        ref = generate_chatbot_ref(db, project_id=config.project_id)

        pattern = (
            rf"{re.escape(CHATBOT_REF_PREFIX)}[0-9a-f]{{{CHATBOT_REF_SUFFIX_LENGTH}}}"
        )
        assert re.fullmatch(pattern, ref)

    def test_small_suffix_is_zero_padded(
        self, db: Session, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _script_randbelow(monkeypatch, [FRESH_SUFFIX])

        assert generate_chatbot_ref(db, project_id=config.project_id) == FRESH_REF

    def test_skips_ref_already_used_in_project(
        self, db: Session, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create_test_chatbot_session(db, config=config, chatbot_ref=USED_REF)
        _script_randbelow(monkeypatch, [USED_SUFFIX, FRESH_SUFFIX])

        assert generate_chatbot_ref(db, project_id=config.project_id) == FRESH_REF

    def test_accepts_ref_used_only_in_another_project(
        self, db: Session, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        create_test_chatbot_session(db, config=other_config, chatbot_ref=USED_REF)
        _script_randbelow(monkeypatch, [USED_SUFFIX])

        assert generate_chatbot_ref(db, project_id=config.project_id) == USED_REF

    def test_raises_and_logs_after_max_attempts(
        self,
        db: Session,
        config: Config,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        create_test_chatbot_session(db, config=config, chatbot_ref=USED_REF)
        # One extra value: a loop that over-runs the cap would return FRESH_REF.
        bounds = _script_randbelow(
            monkeypatch, [USED_SUFFIX] * CHATBOT_REF_MAX_ATTEMPTS + [FRESH_SUFFIX]
        )

        with caplog.at_level(logging.ERROR, logger=chatbot_ref_module.__name__):
            with pytest.raises(ChatbotRefGenerationError) as exc_info:
                generate_chatbot_ref(db, project_id=config.project_id)

        assert len(bounds) == CHATBOT_REF_MAX_ATTEMPTS
        assert "(code: ChatbotRefGenerationError)" in str(exc_info.value)
        [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert record.getMessage().startswith("[generate_chatbot_ref]")
        assert f"project_id: {config.project_id}" in record.getMessage()
