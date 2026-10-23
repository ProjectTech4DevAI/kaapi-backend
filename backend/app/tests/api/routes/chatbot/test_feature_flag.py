from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.tests.api.routes.chatbot.conftest import set_chatbot_flag
from app.tests.utils.auth import TestAuthContext

CHATBOT_URL = f"{settings.API_V1_STR}/chatbot"


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/message", {"json": {}}),
        ("post", "/sessions", {"json": {}}),
        ("get", "/sessions", {"params": {"contact_ref": "c", "chatbot_ref": "b"}}),
        ("post", f"/sessions/{uuid4()}/message", {"json": {}}),
    ],
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
