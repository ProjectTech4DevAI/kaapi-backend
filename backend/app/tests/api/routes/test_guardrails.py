from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.crud import JobCrud
from app.models import Job, JobStatus, JobType, JobUpdate
from app.tests.utils.auth import TestAuthContext


VALIDATOR_ID = str(uuid4())


def _payload(**overrides):
    body = {
        "text": "My email is alice@example.com",
        "config": [{"validator_config_id": VALIDATOR_ID}],
    }
    body.update(overrides)
    return body


# ---------- POST /guardrails ----------


def test_apply_guardrails_poll_variant_returns_job_id(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    with patch("app.api.routes.guardrails.start_job") as mock_start:
        mock_start.return_value = _stub_job()
        resp = client.post(
            "api/v1/guardrails", json=_payload(), headers=user_api_key_header
        )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["job_id"]
    assert "poll" in data["message"].lower()
    mock_start.assert_called_once()


def test_apply_guardrails_callback_variant_message(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    with patch("app.api.routes.guardrails.start_job") as mock_start, patch(
        "app.api.routes.guardrails.validate_callback_url"
    ) as mock_validate:
        mock_start.return_value = _stub_job()
        resp = client.post(
            "api/v1/guardrails",
            json=_payload(callback_url="https://example.com/cb"),
            headers=user_api_key_header,
        )

    assert resp.status_code == 200
    assert "callback" in resp.json()["data"]["message"].lower()
    mock_validate.assert_called_once_with("https://example.com/cb")


def test_apply_guardrails_invalid_callback_url_rejected(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "api/v1/guardrails",
        json=_payload(callback_url="not-a-url"),
        headers=user_api_key_header,
    )
    assert resp.status_code == 422


def test_apply_guardrails_empty_text_422(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "api/v1/guardrails", json=_payload(text=""), headers=user_api_key_header
    )
    assert resp.status_code == 422


def test_apply_guardrails_empty_config_422(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "api/v1/guardrails", json=_payload(config=[]), headers=user_api_key_header
    )
    assert resp.status_code == 422


def test_apply_guardrails_malformed_validator_id_422(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "api/v1/guardrails",
        json=_payload(config=[{"validator_config_id": "not-a-uuid"}]),
        headers=user_api_key_header,
    )
    assert resp.status_code == 422


def test_apply_guardrails_requires_auth(client: TestClient) -> None:
    resp = client.post("api/v1/guardrails", json=_payload())
    assert resp.status_code in (401, 403)


# ---------- GET /guardrails/{job_id} ----------


def test_get_guardrails_unknown_job_404(
    client: TestClient, user_api_key_header: dict[str, str]
) -> None:
    resp = client.get(f"api/v1/guardrails/{uuid4()}", headers=user_api_key_header)
    assert resp.status_code == 404


def test_get_guardrails_non_guardrails_job_404(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
    user_api_key_header: dict[str, str],
) -> None:
    """A non-guardrails job id must 404, not leak existence."""
    job = JobCrud(session=db).create(
        job_type=JobType.LLM_API, project_id=user_api_key.project_id
    )
    resp = client.get(f"api/v1/guardrails/{job.id}", headers=user_api_key_header)
    assert resp.status_code == 404


def test_get_guardrails_success_rehydrates_safe_text(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
    user_api_key_header: dict[str, str],
) -> None:
    safe_text = "My email is [REDACTED]"
    job = _seed_guardrails_job(
        db,
        project_id=user_api_key.project_id,
        status=JobStatus.SUCCESS,
        meta={
            "request": {"text": "My email is alice@example.com"},
            "response": {
                "data": {
                    "safe_text": safe_text,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            },
            "callback": {
                "response_id": "resp-abc",
                "delivered": False,
                "warnings": ["dup ignored"],
            },
        },
    )

    resp = client.get(f"api/v1/guardrails/{job.id}", headers=user_api_key_header)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == JobStatus.SUCCESS.value
    assert data["warnings"] == ["dup ignored"]
    assert (
        data["guardrails_response"]["response"]["output"]["content"]["value"]
        == safe_text
    )
    assert data["guardrails_response"]["response"]["response_id"] == "resp-abc"


def test_get_guardrails_success_falls_back_to_original_text(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
    user_api_key_header: dict[str, str],
) -> None:
    """If upstream omitted safe_text, original request.text is echoed back."""
    job = _seed_guardrails_job(
        db,
        project_id=user_api_key.project_id,
        status=JobStatus.SUCCESS,
        meta={
            "request": {"text": "hello world"},
            "response": {"data": {}},
            "callback": {"response_id": None, "delivered": False, "warnings": []},
        },
    )

    resp = client.get(f"api/v1/guardrails/{job.id}", headers=user_api_key_header)
    assert resp.status_code == 200
    value = resp.json()["data"]["guardrails_response"]["response"]["output"]["content"][
        "value"
    ]
    assert value == "hello world"


def test_get_guardrails_failed_returns_error_message(
    client: TestClient,
    db: Session,
    user_api_key: TestAuthContext,
    user_api_key_header: dict[str, str],
) -> None:
    job = _seed_guardrails_job(
        db,
        project_id=user_api_key.project_id,
        status=JobStatus.FAILED,
        meta={"request": {"text": "hi"}},
        error_message="hard-blocked by guardrails",
    )

    resp = client.get(f"api/v1/guardrails/{job.id}", headers=user_api_key_header)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == JobStatus.FAILED.value
    assert data["error_message"] == "hard-blocked by guardrails"
    assert data["guardrails_response"] is None


# ---------- management-API proxy routes ----------


class TestProxyPassthrough:
    @pytest.mark.parametrize(
        "status_code, body",
        [
            (200, {"success": True, "data": [{"name": "pii"}]}),
            (422, {"detail": [{"loc": ["body", "name"], "msg": "field required"}]}),
            (404, {"detail": "Ban list not found"}),
        ],
    )
    def test_upstream_status_and_body_echoed(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        status_code: int,
        body: dict[str, Any],
    ) -> None:
        with _mock_upstream(status_code=status_code, json_body=body):
            resp = client.get("api/v1/guardrails", headers=user_api_key_header)

        assert resp.status_code == status_code
        assert resp.json() == body

    def test_empty_upstream_body_returns_status_with_no_body(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(status_code=204, content=b""):
            resp = client.delete(
                f"api/v1/guardrails/ban_lists/{uuid4()}", headers=user_api_key_header
            )

        assert resp.status_code == 204
        assert resp.content == b""

    def test_create_echoes_upstream_status_and_body(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        created = {"id": str(uuid4()), "validator_name": "toxicity"}
        # Upstream create routes return FastAPI's default 200.
        with _mock_upstream(status_code=200, json_body=created) as calls:
            resp = client.post(
                "api/v1/guardrails/llm_prompt_configs",
                json={"validator_name": "toxicity", "prompt": "be nice"},
                headers=user_api_key_header,
            )

        assert resp.status_code == 200
        assert resp.json() == created
        assert calls[0]["kwargs"]["json"] == {
            "validator_name": "toxicity",
            "prompt": "be nice",
        }

    def test_connect_error_returns_502(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(raises=httpx.ConnectError("connection refused")):
            resp = client.get("api/v1/guardrails", headers=user_api_key_header)

        assert resp.status_code == 502
        assert resp.json()["error"] == "Guardrails service unavailable"

    def test_non_json_upstream_body_returns_502(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(status_code=200, content=b"<html>gateway</html>"):
            resp = client.get("api/v1/guardrails", headers=user_api_key_header)

        assert resp.status_code == 502
        assert resp.json()["error"] == "Guardrails service returned an invalid response"


class TestProxyForwardedRequest:
    def test_unset_limit_is_dropped_from_forwarded_params(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get("api/v1/guardrails/ban_lists", headers=user_api_key_header)

        params = calls[0]["kwargs"]["params"]
        assert params == {"offset": 0}

    def test_ids_forwarded_as_list(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        first, second = str(uuid4()), str(uuid4())
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get(
                f"api/v1/guardrails/validators/configs?ids={first}&ids={second}"
                "&stage=input",
                headers=user_api_key_header,
            )

        assert calls[0]["kwargs"]["params"] == {
            "ids": [first, second],
            "stage": "input",
        }

    def test_tenant_headers_come_from_auth_context_not_request(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        with _mock_upstream(status_code=200, json_body={"id": str(uuid4())}) as calls:
            client.post(
                "api/v1/guardrails/ban_lists?organization_id=999",
                json={"name": "slurs", "organization_id": 999, "project_id": 888},
                headers=user_api_key_header,
            )

        headers = calls[0]["kwargs"]["headers"]
        assert headers["X-ORGANIZATION-ID"] == str(user_api_key.organization_id)
        assert headers["X-PROJECT-ID"] == str(user_api_key.project_id)

    def test_ban_list_detail_path_forwarded(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        ban_list_id = uuid4()
        with _mock_upstream(json_body={"id": str(ban_list_id)}) as calls:
            resp = client.patch(
                f"api/v1/guardrails/ban_lists/{ban_list_id}",
                json={"name": "renamed"},
                headers=user_api_key_header,
            )

        assert resp.status_code == 200
        method, url = calls[0]["args"]
        assert method == "PATCH"
        assert url.endswith(f"/ban_lists/{ban_list_id}")


class TestProxyRouteOrdering:
    def test_ban_lists_list_route_wins_over_job_status_route(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(json_body={"data": []}) as calls:
            resp = client.get(
                "api/v1/guardrails/ban_lists", headers=user_api_key_header
            )

        assert resp.status_code == 200
        assert calls[0]["args"][1].endswith("/ban_lists/")


def test_list_ban_lists_requires_auth(client: TestClient) -> None:
    resp = client.get("api/v1/guardrails/ban_lists")
    assert resp.status_code in (401, 403)


# ---------- helpers ----------


@contextmanager
def _mock_upstream(
    *,
    status_code: int = 200,
    json_body: Any = None,
    content: bytes | None = None,
    raises: Exception | None = None,
):
    """Stub the guardrails HTTP boundary; yields the recorded client.request calls."""
    calls: list[dict[str, Any]] = []

    response = MagicMock()
    response.status_code = status_code
    if content is None:
        import json as _json

        response.content = _json.dumps(json_body).encode()
        response.json.return_value = json_body
    else:
        response.content = content
        response.json.side_effect = ValueError("not json")

    def _request(*args: Any, **kwargs: Any):
        calls.append({"args": args, "kwargs": kwargs})
        if raises is not None:
            raise raises
        return response

    client = MagicMock()
    client.request.side_effect = _request

    with patch("app.services.llm.guardrails.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = client
        yield calls


def _stub_job() -> Job:
    """Minimal in-memory Job for routes that only read id/status/timestamps."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return Job(
        id=uuid4(),
        job_type=JobType.LLM_GUARDRAILS,
        status=JobStatus.PENDING,
        inserted_at=now,
        updated_at=now,
    )


def _seed_guardrails_job(
    db: Session,
    *,
    project_id: int,
    status: JobStatus,
    meta: dict,
    error_message: str | None = None,
) -> Job:
    crud = JobCrud(session=db)
    job = crud.create(job_type=JobType.LLM_GUARDRAILS, project_id=project_id, meta=meta)
    return crud.update(
        job_id=job.id,
        job_update=JobUpdate(status=status, error_message=error_message),
    )
