from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.crud import JobCrud
from app.models import Job, JobStatus, JobType, JobUpdate
from app.tests.utils.auth import TestAuthContext


VALIDATOR_ID = str(uuid4())

# (client method, kaapi path, request body, upstream path) — "{id}" is filled
# with a freshly generated UUID by the tests that consume this table.
PROXY_ROUTES = [
    ("GET", "/guardrails", None, "/"),
    ("POST", "/guardrails/ban_lists", {"name": "slurs"}, "/ban_lists/"),
    ("GET", "/guardrails/ban_lists", None, "/ban_lists/"),
    ("GET", "/guardrails/ban_lists/{id}", None, "/ban_lists/{id}"),
    ("PATCH", "/guardrails/ban_lists/{id}", {"name": "renamed"}, "/ban_lists/{id}"),
    ("DELETE", "/guardrails/ban_lists/{id}", None, "/ban_lists/{id}"),
    (
        "POST",
        "/guardrails/llm_prompt_configs",
        {"validator_name": "toxicity", "prompt": "be nice"},
        "/llm_prompt_configs/",
    ),
    ("GET", "/guardrails/llm_prompt_configs", None, "/llm_prompt_configs/"),
    (
        "GET",
        "/guardrails/llm_prompt_configs/{id}",
        None,
        "/llm_prompt_configs/{id}",
    ),
    (
        "PATCH",
        "/guardrails/llm_prompt_configs/{id}",
        {"prompt": "be nicer"},
        "/llm_prompt_configs/{id}",
    ),
    (
        "DELETE",
        "/guardrails/llm_prompt_configs/{id}",
        None,
        "/llm_prompt_configs/{id}",
    ),
    (
        "POST",
        "/guardrails/validators/configs",
        {"type": "pii", "stage": "input"},
        "/validators/configs/",
    ),
    ("GET", "/guardrails/validators/configs", None, "/validators/configs/"),
    (
        "GET",
        "/guardrails/validators/configs/{id}",
        None,
        "/validators/configs/{id}",
    ),
    (
        "PATCH",
        "/guardrails/validators/configs/{id}",
        {"stage": "output"},
        "/validators/configs/{id}",
    ),
    (
        "DELETE",
        "/guardrails/validators/configs/{id}",
        None,
        "/validators/configs/{id}",
    ),
]

PROXY_ROUTE_IDS = [f"{method} {path}" for method, path, _, _ in PROXY_ROUTES]


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
    @pytest.mark.parametrize(
        "method, kaapi_path, body, upstream_path", PROXY_ROUTES, ids=PROXY_ROUTE_IDS
    )
    def test_route_forwards_method_path_body_and_tenant(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        method: str,
        kaapi_path: str,
        body: dict[str, Any] | None,
        upstream_path: str,
    ) -> None:
        resource_id = uuid4()
        upstream_body = {"ok": True}
        with _mock_upstream(json_body=upstream_body) as calls:
            resp = client.request(
                method,
                f"api/v1{kaapi_path.format(id=resource_id)}",
                json=body,
                headers=user_api_key_header,
            )

        assert resp.status_code == 200
        assert resp.json() == upstream_body
        assert calls[0]["args"] == (
            method,
            f"{settings.KAAPI_GUARDRAILS_URL}{upstream_path.format(id=resource_id)}",
        )
        assert calls[0]["kwargs"]["json"] == body
        headers = calls[0]["kwargs"]["headers"]
        assert headers["X-ORGANIZATION-ID"] == str(user_api_key.organization_id)
        assert headers["X-PROJECT-ID"] == str(user_api_key.project_id)

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

    def test_ban_list_filters_forwarded(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get(
                "api/v1/guardrails/ban_lists?domain=email&offset=5&limit=10",
                headers=user_api_key_header,
            )

        assert calls[0]["kwargs"]["params"] == {
            "domain": "email",
            "offset": 5,
            "limit": 10,
        }

    def test_llm_prompt_config_filters_forwarded(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get(
                "api/v1/guardrails/llm_prompt_configs?validator_name=toxicity&limit=50",
                headers=user_api_key_header,
            )

        assert calls[0]["kwargs"]["params"] == {
            "validator_name": "toxicity",
            "offset": 0,
            "limit": 50,
        }

    def test_validator_config_filters_forwarded_without_ids(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get(
                "api/v1/guardrails/validators/configs?stage=output&type=pii",
                headers=user_api_key_header,
            )

        assert calls[0]["kwargs"]["params"] == {"stage": "output", "type": "pii"}

    def test_ids_are_normalised_to_canonical_uuid_strings(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        """Upstream expects strings; UUID parsing also canonicalises the casing."""
        config_id = uuid4()
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get(
                f"api/v1/guardrails/validators/configs?ids={str(config_id).upper()}",
                headers=user_api_key_header,
            )

        assert calls[0]["kwargs"]["params"] == {"ids": [str(config_id)]}

    @pytest.mark.parametrize(
        "kaapi_path", ["/guardrails/ban_lists", "/guardrails/llm_prompt_configs"]
    )
    @pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
    def test_out_of_range_pagination_rejected(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        kaapi_path: str,
        query: str,
    ) -> None:
        resp = client.get(f"api/v1{kaapi_path}?{query}", headers=user_api_key_header)
        assert resp.status_code == 422

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

    def test_validator_configs_list_and_detail_routes_do_not_collide(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        config_id = uuid4()
        with _mock_upstream(json_body={"data": []}) as calls:
            client.get(
                "api/v1/guardrails/validators/configs", headers=user_api_key_header
            )
            client.get(
                f"api/v1/guardrails/validators/configs/{config_id}",
                headers=user_api_key_header,
            )

        assert calls[0]["args"][1].endswith("/validators/configs/")
        assert calls[1]["args"][1].endswith(f"/validators/configs/{config_id}")

    def test_llm_prompt_configs_list_route_wins_over_job_status_route(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        with _mock_upstream(json_body={"data": []}) as calls:
            resp = client.get(
                "api/v1/guardrails/llm_prompt_configs", headers=user_api_key_header
            )

        assert resp.status_code == 200
        assert calls[0]["args"][1].endswith("/llm_prompt_configs/")


def test_list_ban_lists_requires_auth(client: TestClient) -> None:
    resp = client.get("api/v1/guardrails/ban_lists")
    assert resp.status_code in (401, 403)


@pytest.mark.parametrize(
    "method, kaapi_path, body, _upstream_path", PROXY_ROUTES, ids=PROXY_ROUTE_IDS
)
def test_proxy_routes_require_auth(
    client: TestClient,
    method: str,
    kaapi_path: str,
    body: dict[str, Any] | None,
    _upstream_path: str,
) -> None:
    with _mock_upstream(json_body={"data": []}) as calls:
        resp = client.request(
            method, f"api/v1{kaapi_path.format(id=uuid4())}", json=body
        )

    assert resp.status_code in (401, 403)
    assert calls == []


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
