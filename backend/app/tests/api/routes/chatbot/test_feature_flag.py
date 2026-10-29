from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.tests.api.routes.chatbot.conftest import set_chatbot_flag
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import CHATBOTS_URL, UNKNOWN_CHATBOT_ID

UNKNOWN_SESSION_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "", {"json": {"name": "bot"}}),
        ("get", "", {}),
        ("get", f"/{UNKNOWN_CHATBOT_ID}", {}),
        ("patch", f"/{UNKNOWN_CHATBOT_ID}", {"json": {"name": "bot"}}),
        ("delete", f"/{UNKNOWN_CHATBOT_ID}", {}),
        ("get", f"/{UNKNOWN_CHATBOT_ID}/flow", {}),
        ("put", f"/{UNKNOWN_CHATBOT_ID}/flow", {"json": {"nodes": []}}),
        ("post", f"/{UNKNOWN_CHATBOT_ID}/flow/publish", {}),
        ("post", f"/{UNKNOWN_CHATBOT_ID}/sessions", {"json": {"contact_ref": "c"}}),
        ("get", "/sessions/latest", {"params": {"contact_ref": "c"}}),
        ("post", f"/sessions/{UNKNOWN_SESSION_ID}/message", {"json": {}}),
    ],
    ids=[
        "create_chatbot",
        "list_chatbots",
        "get_chatbot",
        "update_chatbot",
        "delete_chatbot",
        "get_flow",
        "save_flow",
        "publish_flow",
        "create_session",
        "get_latest_session",
        "send_message",
    ],
)
def test_chatbot_routes_return_403_when_flag_disabled(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
    user_api_key_header: dict[str, str],
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    set_chatbot_flag(db, user_api_key, enabled=False)

    resp = client.request(
        method,
        f"{CHATBOTS_URL}{path}",
        headers=user_api_key_header,
        **kwargs,
    )

    assert resp.status_code == 403
    assert "CHATBOT" in resp.json()["error"]
