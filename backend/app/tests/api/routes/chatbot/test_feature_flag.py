from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.api.routes.chatbot.conftest import set_chatbot_flag
from app.tests.utils.auth import TestAuthContext

CHATBOT_URL = f"{settings.API_V1_STR}/chatbot"
UNKNOWN_SESSION_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/sessions", {"json": {}}),
        ("get", "/sessions", {"params": {"contact_ref": "c", "chatbot_ref": "b"}}),
        ("post", f"/sessions/{UNKNOWN_SESSION_ID}/message", {"json": {}}),
    ],
    ids=["create_session", "get_latest_session", "send_message"],
)
def test_chatbot_routes_return_403_when_flag_disabled(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    set_chatbot_flag(db, user_api_key, enabled=False)

    resp = client.request(
        method,
        f"{CHATBOT_URL}{path}",
        headers={"X-API-KEY": user_api_key.key},
        **kwargs,
    )

    assert resp.status_code == 403
    assert "CHATBOT" in resp.json()["error"]
