import pytest
from sqlmodel import Session

from app.core.feature_flags import FeatureFlag
from app.crud.feature_flag import (
    create_feature_flag,
    get_feature_flag,
    update_feature_flag,
)
from app.models import Chatbot, Config
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import (
    create_test_anthropic_config,
    create_test_chatbot,
    create_test_live_chatbot,
)


def set_chatbot_flag(db: Session, auth: TestAuthContext, *, enabled: bool) -> None:
    scope = {
        "session": db,
        "key": FeatureFlag.CHATBOT.value,
        "organization_id": auth.organization_id,
        "project_id": auth.project_id,
    }
    if get_feature_flag(**scope) is None:
        create_feature_flag(**scope, enabled=enabled)
    else:
        update_feature_flag(**scope, enabled=enabled)


@pytest.fixture(autouse=True)
def chatbot_feature_enabled(db: Session, user_api_key: TestAuthContext) -> None:
    set_chatbot_flag(db, user_api_key, enabled=True)


@pytest.fixture
def config(db: Session, user_api_key: TestAuthContext) -> Config:
    return create_test_anthropic_config(db, project_id=user_api_key.project_id)


@pytest.fixture
def chatbot(db: Session, user_api_key: TestAuthContext) -> Chatbot:
    """A chatbot in the caller's project with no flow saved yet."""
    return create_test_chatbot(db, project=user_api_key.project)


@pytest.fixture
def live_chatbot(db: Session, config: Config) -> Chatbot:
    """A chatbot in the caller's project whose live flow is one onboarding node."""
    return create_test_live_chatbot(db, config=config)[0]


@pytest.fixture
def foreign_chatbot(db: Session) -> Chatbot:
    return create_test_chatbot(db)


@pytest.fixture
def foreign_live_chatbot(db: Session) -> Chatbot:
    return create_test_live_chatbot(db, config=create_test_anthropic_config(db))[0]
