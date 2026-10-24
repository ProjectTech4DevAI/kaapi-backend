import pytest
from sqlmodel import Session

from app.core.feature_flags import FeatureFlag
from app.crud.feature_flag import (
    create_feature_flag,
    get_feature_flag,
    update_feature_flag,
)
from app.tests.utils.auth import TestAuthContext


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
