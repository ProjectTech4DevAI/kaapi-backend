import json

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.routes.agent import _extract_forwardable_credentials
from app.core.config import settings
from app.tests.utils.agent import (
    anthropic_message,
    install_fake_anthropic,
    project_access_token,
    text_message,
    tool_use_message,
)
from app.services.agent.tools import READ_ONLY_TOOLS
from app.services.agent.prompts import AGENT_EMPTY_ANSWER_MESSAGE, AGENT_REFUSAL_MESSAGE
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config, create_test_evaluation_run

AGENT_URL = f"{settings.API_V1_STR}/agent"


class TestAgentQueryEndToEnd:
    def test_lists_seeded_eval_runs_and_answers(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        run = create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        fake = install_fake_anthropic(
            monkeypatch,
            [
                tool_use_message("list_evaluation_runs", {"limit": 5}),
                text_message(f"You have one run: {run.run_name}."),
            ],
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "List my runs"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["answer"] == f"You have one run: {run.run_name}."
        assert body["stop_reason"] == "completed"
        assert body["iterations"] == 1
        assert body["usage"] == {"input_tokens": 20, "output_tokens": 10}
        assert len(body["tool_calls"]) == 1
        tool_call = body["tool_calls"][0]
        assert tool_call["name"] == "list_evaluation_runs"
        assert tool_call["status_code"] == 200
        assert tool_call["is_error"] is False
        assert tool_call["arguments"] == {"limit": 5, "offset": 0}

        assert len(fake.calls) == 2
        [tool_result] = fake.tool_results(1)
        assert tool_result["tool_use_id"] == "toolu_1"
        assert tool_result["is_error"] is False
        runs = json.loads(tool_result["content"])
        assert [r["id"] for r in runs] == [run.id]
        assert runs[0]["run_name"] == run.run_name

    def test_eval_run_projection_hides_traces_and_storage_urls(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        run = create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            score={
                "summary_scores": [{"name": "accuracy", "avg": 0.75}],
                "traces": [{"trace_id": "trace-secret-1", "question": "Q1"}],
            },
            object_store_url="s3://internal-bucket/run.csv",
        )
        fake = install_fake_anthropic(
            monkeypatch,
            [
                tool_use_message("get_evaluation_run", {"evaluation_id": run.id}),
                text_message("Accuracy 0.75."),
            ],
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Score of my run?"}
        )

        assert resp.status_code == 200
        [tool_result] = fake.tool_results(1)
        projected = json.loads(tool_result["content"])
        assert projected["id"] == run.id
        assert projected["score"] == {
            "summary_scores": [{"name": "accuracy", "avg": 0.75}]
        }
        assert "trace-secret-1" not in fake.sent_payload()
        assert "internal-bucket" not in fake.sent_payload()

    @pytest.mark.parametrize(
        "tool_name",
        [
            tool.name
            for tool in READ_ONLY_TOOLS
            if not any(f.is_required() for f in tool.args_model.model_fields.values())
        ],
    )
    def test_every_argless_tool_reaches_its_real_route(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
        tool_name: str,
    ) -> None:
        fake = install_fake_anthropic(
            monkeypatch, [tool_use_message(tool_name, {}), text_message("ok")]
        )

        resp = client.post(AGENT_URL, headers=user_api_key_header, json={"query": "q"})

        assert resp.status_code == 200
        [tool_call] = resp.json()["data"]["tool_calls"]
        assert tool_call["status_code"] == 200, fake.tool_results(1)[0]["content"]


class TestAgentTenantIsolation:
    def test_other_projects_eval_run_is_not_visible(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        superuser_api_key: TestAuthContext,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        assert superuser_api_key.project_id != user_api_key.project_id
        other_run = create_test_evaluation_run(
            db,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
        )
        fake = install_fake_anthropic(
            monkeypatch,
            [
                # Separate turns: parallel inner requests would share the test's
                # single DB session across threads, which SQLAlchemy doesn't allow.
                tool_use_message(
                    "get_evaluation_run",
                    {"evaluation_id": other_run.id},
                    tool_use_id="toolu_get",
                ),
                tool_use_message(
                    "list_evaluation_runs", {"limit": 100}, tool_use_id="toolu_list"
                ),
                text_message("Not found."),
            ],
        )

        resp = client.post(
            AGENT_URL,
            headers=user_api_key_header,
            json={"query": f"Show eval run {other_run.id}"},
        )

        assert resp.status_code == 200
        get_call, list_call = resp.json()["data"]["tool_calls"]
        assert get_call["name"] == "get_evaluation_run"
        assert get_call["is_error"] is True
        assert get_call["status_code"] == 404
        assert list_call["status_code"] == 200

        [get_result] = fake.tool_results(1)
        [list_result] = fake.tool_results(2)
        assert get_result["tool_use_id"] == "toolu_get"
        assert get_result["is_error"] is True
        assert json.loads(get_result["content"])["status_code"] == 404
        assert list_result["tool_use_id"] == "toolu_list"
        listed_ids = [r["id"] for r in json.loads(list_result["content"])]
        assert other_run.id not in listed_ids
        assert other_run.run_name not in fake.sent_payload()

    def test_config_version_scoped_to_caller_project(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        superuser_api_key: TestAuthContext,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        own_config = create_test_config(db, project_id=user_api_key.project_id)
        other_config = create_test_config(db, project_id=superuser_api_key.project_id)
        fake = install_fake_anthropic(
            monkeypatch,
            [
                tool_use_message(
                    "get_config_version",
                    {"config_id": str(own_config.id), "version_number": 1},
                    tool_use_id="toolu_own",
                ),
                tool_use_message(
                    "get_config_version",
                    {"config_id": str(other_config.id), "version_number": 1},
                    tool_use_id="toolu_other",
                ),
                text_message("done"),
            ],
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Show configs"}
        )

        assert resp.status_code == 200
        own_call, other_call = resp.json()["data"]["tool_calls"]
        assert own_call["status_code"] == 200
        assert other_call["status_code"] == 404
        [own_result] = fake.tool_results(1)
        assert json.loads(own_result["content"])["config_id"] == str(own_config.id)
        assert other_config.name not in fake.sent_payload()


class TestAgentCredentialSecrecy:
    @pytest.mark.parametrize("channel", ["api_key", "bearer", "cookie"])
    def test_credential_never_reaches_model_or_response(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        channel: str,
    ) -> None:
        create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        headers: dict[str, str] = {}
        if channel == "api_key":
            secret = user_api_key.key
            headers["X-API-KEY"] = secret
        elif channel == "bearer":
            secret = project_access_token(user_api_key)
            headers["Authorization"] = f"Bearer {secret}"
        else:
            secret = project_access_token(user_api_key)
            client.cookies.set("access_token", secret)
        fake = install_fake_anthropic(
            monkeypatch,
            [tool_use_message("list_evaluation_runs", {}), text_message("Done.")],
        )

        resp = client.post(AGENT_URL, headers=headers, json={"query": "List runs"})

        assert resp.status_code == 200
        # The inner call must have authenticated, or "secret absent" proves nothing.
        assert resp.json()["data"]["tool_calls"][0]["status_code"] == 200
        # Strip the "ApiKey " prefix too: the bare key must not leak either.
        bare_secret = secret.removeprefix("ApiKey ")
        assert bare_secret not in fake.sent_payload()
        assert bare_secret not in resp.text


def _anthropic_http_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        json={"type": "error", "error": {"message": "upstream said no"}},
    )


class TestAgentStopReasons:
    def test_iteration_cap_disables_tools_and_reports_limit(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        monkeypatch.setattr(settings, "AGENT_MAX_ITERATIONS", 1)
        # A single script entry is replayed forever: a model that never stops asking.
        fake = install_fake_anthropic(
            monkeypatch, [tool_use_message("list_evaluation_runs", {})]
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Loop forever"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["stop_reason"] == "iteration_limit"
        assert body["iterations"] == 1
        assert body["answer"] == AGENT_EMPTY_ANSWER_MESSAGE
        assert len(body["tool_calls"]) == 1
        assert [call["tool_choice"] for call in fake.calls] == [
            {"type": "auto"},
            {"type": "none"},
        ]
        # Tools stay in the request even when disabled so the cached prefix is stable.
        assert fake.calls[1]["tools"] == fake.calls[0]["tools"]

    def test_refusal_without_text_returns_fixed_decline(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        install_fake_anthropic(monkeypatch, [anthropic_message([], "refusal")])

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Do something bad"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["stop_reason"] == "refusal"
        assert body["answer"] == AGENT_REFUSAL_MESSAGE
        assert body["tool_calls"] == []

    def test_max_tokens_keeps_partial_text(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        install_fake_anthropic(
            monkeypatch, [text_message("Partial answer", stop_reason="max_tokens")]
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Long question"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["stop_reason"] == "max_tokens"
        assert body["answer"] == "Partial answer"


class TestAgentErrors:
    def test_missing_anthropic_key_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        fake = install_fake_anthropic(monkeypatch, [text_message("unused")])
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Anything"}
        )

        assert resp.status_code == 503
        assert "not configured" in resp.json()["error"]
        assert fake.calls == []

    @pytest.mark.parametrize(
        "exc, expected_fragment",
        [
            (
                anthropic.APIStatusError(
                    "Error code: 500",
                    response=_anthropic_http_response(500),
                    body=None,
                ),
                "code: 500",
            ),
            (
                anthropic.RateLimitError(
                    "Error code: 429",
                    response=_anthropic_http_response(429),
                    body=None,
                ),
                "Rate limit exceeded",
            ),
        ],
        ids=["status_error", "rate_limit"],
    )
    def test_anthropic_failure_returns_502(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
        exc: anthropic.APIError,
        expected_fragment: str,
    ) -> None:
        install_fake_anthropic(monkeypatch, [exc])

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Anything"}
        )

        assert resp.status_code == 502
        error = resp.json()["error"]
        assert "[ANTHROPIC]" in error
        assert expected_fragment in error
        # Upstream bodies can carry request internals; only the curated message is shown.
        assert "upstream said no" not in error

    def test_requires_authentication(self, client: TestClient) -> None:
        resp = client.post(AGENT_URL, json={"query": "Anything"})

        assert resp.status_code == 401

    def test_empty_query_rejected(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        resp = client.post(AGENT_URL, headers=user_api_key_header, json={"query": ""})

        assert resp.status_code == 422


class TestExtractForwardableCredentials:
    @pytest.mark.parametrize(
        "api_key, bearer_token, cookie_token, expected",
        [
            ("ApiKey abc", None, None, {"X-API-KEY": "ApiKey abc"}),
            (None, "tok", None, {"Authorization": "Bearer tok"}),
            (None, None, "cookietok", {"Authorization": "Bearer cookietok"}),
            (
                "ApiKey abc",
                "tok",
                None,
                {"X-API-KEY": "ApiKey abc", "Authorization": "Bearer tok"},
            ),
            (None, "tok", "cookietok", {"Authorization": "Bearer tok"}),
            (None, None, None, {}),
        ],
        ids=[
            "api_key_only",
            "bearer_only",
            "cookie_only",
            "api_key_and_bearer",
            "bearer_beats_cookie",
            "nothing_forwardable",
        ],
    )
    def test_extracts_in_auth_priority_order(
        self,
        api_key: str | None,
        bearer_token: str | None,
        cookie_token: str | None,
        expected: dict[str, str],
    ) -> None:
        assert (
            _extract_forwardable_credentials(
                api_key=api_key,
                bearer_token=bearer_token,
                cookie_token=cookie_token,
            )
            == expected
        )
