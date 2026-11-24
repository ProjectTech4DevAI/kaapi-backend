from datetime import timedelta
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.util import now
from app.models import (
    WebhookDeliveryStatusEnum,
    WebhookLog,
    WebhookResourceTypeEnum,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_project

URL = "/api/v1/webhook-logs"


def _make_log(
    db: Session,
    *,
    project_id: int | None,
    organization_id: int | None,
    minutes_ago: int = 0,
    **overrides: Any,
) -> WebhookLog:
    fields: dict[str, Any] = {
        "resource_type": WebhookResourceTypeEnum.LLM_CALL,
        "resource_id": str(uuid4()),
        "url": "https://example.com/hook",
        "status": WebhookDeliveryStatusEnum.DELIVERED,
        "status_code": 200,
        "attempt_count": 1,
        "request_bytes": 12,
        "signed": True,
        # explicit timestamps so ordering doesn't depend on insert timing
        "inserted_at": now() - timedelta(minutes=minutes_ago),
        **overrides,
    }
    log = WebhookLog(project_id=project_id, organization_id=organization_id, **fields)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def _own(db: Session, ctx: TestAuthContext, **kw: Any) -> WebhookLog:
    return _make_log(
        db, project_id=ctx.project_id, organization_id=ctx.organization_id, **kw
    )


def _ids(resp) -> list[str]:
    return [row["id"] for row in resp.json()["data"]]


class TestListWebhookLogs:
    def test_returns_newest_first(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        old = _own(db, user_api_key, minutes_ago=10)
        new = _own(db, user_api_key, minutes_ago=1)
        mid = _own(db, user_api_key, minutes_ago=5)

        resp = client.get(URL, headers={"X-API-KEY": user_api_key.key})

        assert resp.status_code == 200
        assert _ids(resp) == [str(new.id), str(mid.id), str(old.id)]
        assert resp.json()["metadata"] == {"skip": 0, "limit": 100, "total": 3}

    def test_excludes_other_project_other_org_and_null_project_rows(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        mine = _own(db, user_api_key)
        sibling_project = create_test_project(db, user_api_key.organization_id)
        foreign_project = create_test_project(db)
        _make_log(
            db,
            project_id=sibling_project.id,
            organization_id=user_api_key.organization_id,
        )
        _make_log(
            db,
            project_id=foreign_project.id,
            organization_id=foreign_project.organization_id,
        )
        # right project, wrong org: org filter must apply independently
        _make_log(
            db,
            project_id=user_api_key.project_id,
            organization_id=foreign_project.organization_id,
        )
        _make_log(db, project_id=None, organization_id=user_api_key.organization_id)
        _make_log(db, project_id=None, organization_id=None)

        resp = client.get(URL, headers={"X-API-KEY": user_api_key.key})

        assert resp.status_code == 200
        assert _ids(resp) == [str(mine.id)]
        assert resp.json()["metadata"]["total"] == 1

    def test_filter_by_resource_type(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        match = _own(db, user_api_key, resource_type=WebhookResourceTypeEnum.RESPONSE)
        _own(db, user_api_key, resource_type=WebhookResourceTypeEnum.LLM_CALL)

        resp = client.get(
            URL,
            params={"resource_type": "RESPONSE"},
            headers={"X-API-KEY": user_api_key.key},
        )

        assert resp.status_code == 200
        assert _ids(resp) == [str(match.id)]
        assert resp.json()["metadata"]["total"] == 1

    def test_filter_by_resource_id(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        match = _own(db, user_api_key, resource_id="job-123")
        _own(db, user_api_key, resource_id="job-456")

        resp = client.get(
            URL,
            params={"resource_id": "job-123"},
            headers={"X-API-KEY": user_api_key.key},
        )

        assert _ids(resp) == [str(match.id)]

    def test_filter_by_status(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        match = _own(
            db,
            user_api_key,
            status=WebhookDeliveryStatusEnum.FAILED,
            status_code=500,
            error="boom",
        )
        _own(db, user_api_key, status=WebhookDeliveryStatusEnum.DELIVERED)

        resp = client.get(
            URL, params={"status": "FAILED"}, headers={"X-API-KEY": user_api_key.key}
        )

        assert _ids(resp) == [str(match.id)]

    def test_combined_filters_intersect(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        match = _own(
            db,
            user_api_key,
            resource_type=WebhookResourceTypeEnum.COLLECTION,
            resource_id="c-1",
            status=WebhookDeliveryStatusEnum.REJECTED,
            attempt_count=0,
        )
        # each decoy matches all but one filter
        _own(
            db,
            user_api_key,
            resource_type=WebhookResourceTypeEnum.LLM_CALL,
            resource_id="c-1",
            status=WebhookDeliveryStatusEnum.REJECTED,
        )
        _own(
            db,
            user_api_key,
            resource_type=WebhookResourceTypeEnum.COLLECTION,
            resource_id="c-2",
            status=WebhookDeliveryStatusEnum.REJECTED,
        )
        _own(
            db,
            user_api_key,
            resource_type=WebhookResourceTypeEnum.COLLECTION,
            resource_id="c-1",
            status=WebhookDeliveryStatusEnum.DELIVERED,
        )

        resp = client.get(
            URL,
            params={
                "resource_type": "COLLECTION",
                "resource_id": "c-1",
                "status": "REJECTED",
            },
            headers={"X-API-KEY": user_api_key.key},
        )

        assert _ids(resp) == [str(match.id)]
        assert resp.json()["metadata"]["total"] == 1

    def test_skip_and_limit_page_while_total_counts_all(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        logs = [_own(db, user_api_key, minutes_ago=m) for m in range(5)]

        resp = client.get(
            URL,
            params={"skip": 1, "limit": 2},
            headers={"X-API-KEY": user_api_key.key},
        )

        assert resp.status_code == 200
        assert _ids(resp) == [str(logs[1].id), str(logs[2].id)]
        assert resp.json()["metadata"] == {"skip": 1, "limit": 2, "total": 5}

    def test_invalid_enum_filter_returns_422(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        resp = client.get(
            URL, params={"status": "delivered"}, headers={"X-API-KEY": user_api_key.key}
        )
        assert resp.status_code == 422

    def test_limit_out_of_range_returns_422(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        headers = {"X-API-KEY": user_api_key.key}
        assert client.get(URL, params={"limit": 0}, headers=headers).status_code == 422
        assert (
            client.get(URL, params={"limit": 101}, headers=headers).status_code == 422
        )
        assert client.get(URL, params={"skip": -1}, headers=headers).status_code == 422

    def test_caller_without_project_is_forbidden(
        self, client: TestClient, normal_user_token_headers: dict[str, str]
    ) -> None:
        resp = client.get(URL, headers=normal_user_token_headers)
        assert resp.status_code == 403

    def test_unauthenticated_is_rejected(self, client: TestClient) -> None:
        assert client.get(URL).status_code == 401


class TestGetWebhookLog:
    def test_returns_own_row(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        log = _own(
            db,
            user_api_key,
            resource_type=WebhookResourceTypeEnum.EVALUATION,
            resource_id="42",
            status=WebhookDeliveryStatusEnum.FAILED,
            status_code=503,
            attempt_count=3,
            latency=1234.5,
            request_metadata={"conversation_id": "c-42"},
            request_bytes=30,
            response_body="unavailable",
            error="retries exhausted",
            signed=False,
            trace_id="trace-abc",
        )

        resp = client.get(f"{URL}/{log.id}", headers={"X-API-KEY": user_api_key.key})

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["id"] == str(log.id)
        assert body["project_id"] == user_api_key.project_id
        assert body["organization_id"] == user_api_key.organization_id
        assert body["resource_type"] == "EVALUATION"
        assert body["resource_id"] == "42"
        assert body["url"] == "https://example.com/hook"
        assert body["status"] == "FAILED"
        assert body["status_code"] == 503
        assert body["attempt_count"] == 3
        assert body["latency"] == 1234.5
        assert body["request_metadata"] == {"conversation_id": "c-42"}
        assert "request_body" not in body
        assert body["request_bytes"] == 30
        assert body["response_body"] == "unavailable"
        assert body["error"] == "retries exhausted"
        assert body["signed"] is False
        assert body["trace_id"] == "trace-abc"

    def test_unknown_id_returns_404(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        resp = client.get(f"{URL}/{uuid4()}", headers={"X-API-KEY": user_api_key.key})
        assert resp.status_code == 404
        assert resp.json()["error"] == "Webhook log not found"

    def test_other_project_row_returns_404(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        other = create_test_project(db, user_api_key.organization_id)
        log = _make_log(
            db, project_id=other.id, organization_id=user_api_key.organization_id
        )

        resp = client.get(f"{URL}/{log.id}", headers={"X-API-KEY": user_api_key.key})

        assert resp.status_code == 404

    def test_other_org_row_returns_404(
        self, client: TestClient, db: Session, user_api_key: TestAuthContext
    ) -> None:
        foreign = create_test_project(db)
        log = _make_log(
            db,
            project_id=user_api_key.project_id,
            organization_id=foreign.organization_id,
        )

        resp = client.get(f"{URL}/{log.id}", headers={"X-API-KEY": user_api_key.key})

        assert resp.status_code == 404

    def test_non_uuid_id_returns_422(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        resp = client.get(f"{URL}/123", headers={"X-API-KEY": user_api_key.key})
        assert resp.status_code == 422

    def test_caller_without_project_is_forbidden(
        self, client: TestClient, normal_user_token_headers: dict[str, str]
    ) -> None:
        resp = client.get(f"{URL}/{uuid4()}", headers=normal_user_token_headers)
        assert resp.status_code == 403
