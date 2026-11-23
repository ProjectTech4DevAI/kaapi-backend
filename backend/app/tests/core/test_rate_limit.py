"""Tests for the per-organization rate limiter in core/rate_limit.py.

The autouse `isolated_rate_limiter` fixture swaps Redis for an in-memory
moving window, so each test starts with an empty window.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from limits.errors import StorageError

from app.core import rate_limit
from app.core.config import settings

V2_EVALUATIONS_URL = f"{settings.API_V2_STR}/evaluations"
V2_ITERATIONS_URL = f"{settings.API_V2_STR}/evaluations/iterations"
GENERIC_DETAIL = re.compile(r"^Too many requests\. Please try again in \d+ seconds\.$")


def _auth_context(org_id: int = 1, project_id: int | None = 10) -> SimpleNamespace:
    project = None if project_id is None else SimpleNamespace(id=project_id)
    return SimpleNamespace(organization_=SimpleNamespace(id=org_id), project=project)


class TestEnforceRateLimit:
    def test_allows_up_to_limit_then_rejects_with_429(self) -> None:
        enforce = rate_limit.enforce_rate_limit("evaluations")
        limit = settings.EVAL_RATE_LIMIT_PER_MINUTE

        for _ in range(limit):
            enforce(_auth_context())

        with pytest.raises(HTTPException) as exc_info:
            enforce(_auth_context())

        exc = exc_info.value
        assert exc.status_code == 429
        assert 1 <= int(exc.headers["Retry-After"]) <= 60

    def test_rejection_detail_is_generic(self) -> None:
        enforce = rate_limit.enforce_rate_limit("evaluations")
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            enforce(_auth_context())

        with pytest.raises(HTTPException) as exc_info:
            enforce(_auth_context())

        # The API must not leak the category, the limit value or the per-org scope
        assert GENERIC_DETAIL.match(exc_info.value.detail)

    def test_organizations_have_separate_windows(self) -> None:
        enforce = rate_limit.enforce_rate_limit("evaluations")
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            enforce(_auth_context(org_id=1))

        enforce(_auth_context(org_id=2))

    def test_projects_of_one_organization_share_a_window(self) -> None:
        enforce = rate_limit.enforce_rate_limit("evaluations")
        for project_id in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            enforce(_auth_context(org_id=1, project_id=project_id))

        with pytest.raises(HTTPException):
            enforce(_auth_context(org_id=1, project_id=999))

    def test_rejects_without_project_on_auth_context(self) -> None:
        enforce = rate_limit.enforce_rate_limit("evaluations")
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            enforce(_auth_context(project_id=None))

        with pytest.raises(HTTPException) as exc_info:
            enforce(_auth_context(project_id=None))

        assert exc_info.value.status_code == 429

    def test_fails_open_when_redis_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken_limiter = MagicMock()
        broken_limiter.hit.side_effect = StorageError(ConnectionError("redis down"))
        monkeypatch.setattr(rate_limit, "_limiter", broken_limiter)
        enforce = rate_limit.enforce_rate_limit("evaluations")

        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE + 2):
            enforce(_auth_context())

    def test_unknown_category_fails_when_route_is_declared(self) -> None:
        with pytest.raises(KeyError):
            rate_limit.enforce_rate_limit("unknown")  # type: ignore[arg-type]


class TestV2EvaluationRoutesRateLimit:
    # Empty bodies 422 after the limiter has counted the hit, so no dataset/config setup is needed
    def test_v2_evaluation_returns_429_after_limit(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            resp = client.post(V2_EVALUATIONS_URL, headers=user_api_key_header, json={})
            assert resp.status_code == 422

        resp = client.post(V2_EVALUATIONS_URL, headers=user_api_key_header, json={})

        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1
        body = resp.json()
        assert body["success"] is False
        assert GENERIC_DETAIL.match(body["error"])

    def test_v2_evaluation_and_iteration_share_one_bucket(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            client.post(V2_EVALUATIONS_URL, headers=user_api_key_header, json={})

        resp = client.post(V2_ITERATIONS_URL, headers=user_api_key_header, json={})

        assert resp.status_code == 429

    def test_v1_evaluation_is_not_rate_limited(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        url = f"{settings.API_V1_STR}/evaluations"
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE + 1):
            resp = client.post(url, headers=user_api_key_header, json={})
            assert resp.status_code == 422
