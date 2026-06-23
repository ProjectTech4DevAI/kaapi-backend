from unittest.mock import patch
from uuid import uuid4

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


# ---------- helpers ----------


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
