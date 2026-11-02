from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.celery.tasks.job_execution import run_chatbot_session_message
from app.celery.utils import start_chatbot_session_message


class TestStartChatbotSessionMessage:
    def test_publishes_turn_and_returns_task_id(self):
        published: list[dict[str, Any]] = []

        def fake_apply_async(**kwargs: Any) -> SimpleNamespace:
            published.append(kwargs)
            return SimpleNamespace(id="task-123")

        # apply_async is the broker boundary.
        with patch.object(run_chatbot_session_message, "apply_async", fake_apply_async):
            task_id = start_chatbot_session_message(
                session_id="session-1",
                organization_id=2,
                project_id=3,
                webhook_url="https://hooks.example.com/chatbot",
                trace_id="trace-1",
            )

        assert task_id == "task-123"
        [message] = published
        assert message["kwargs"] == {
            "session_id": "session-1",
            "organization_id": 2,
            "project_id": 3,
            "webhook_url": "https://hooks.example.com/chatbot",
            "trace_id": "trace-1",
        }
        assert "otel" in message["headers"]
