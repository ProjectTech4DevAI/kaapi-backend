"""Tests for the per-organization rate limiter in core/rate_limit.py.

The autouse `isolated_rate_limiter` fixture swaps Redis for an in-memory
moving window, so each test starts with an empty window.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from limits.errors import StorageError

from app.core import rate_limit
from app.core.config import settings
from app.core.telemetry import metrics as telemetry

V2_EVALUATIONS_URL = f"{settings.API_V2_STR}/evaluations"
V2_ITERATIONS_URL = f"{settings.API_V2_STR}/evaluations/iterations"
# The run id is never looked up: the empty body 422s right after the limiter counts the hit
V2_IMPROVE_PROMPT_URL = f"{settings.API_V2_STR}/evaluations/1/improve-prompt"
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


class TestRateLimitAlert:
    # Sentry is the external boundary, so the alert helper is patched where rate_limit imports it
    def test_each_rejection_sends_an_alert_with_org_and_project(self) -> None:
        enforce = rate_limit.enforce_rate_limit("evaluations")
        limit = settings.EVAL_RATE_LIMIT_PER_MINUTE

        with patch.object(rate_limit, "record_rate_limit_exceeded") as alert:
            for _ in range(limit):
                enforce(_auth_context(org_id=7, project_id=70))
            assert alert.call_count == 0

            for _ in range(2):
                with pytest.raises(HTTPException):
                    enforce(_auth_context(org_id=7, project_id=70))

        assert alert.call_count == 2
        alert.assert_called_with(
            organization_id=7,
            project_id=70,
            category="evaluations",
            limit_per_minute=limit,
        )

    def test_no_alert_when_redis_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken_limiter = MagicMock()
        broken_limiter.hit.side_effect = StorageError(ConnectionError("redis down"))
        monkeypatch.setattr(rate_limit, "_limiter", broken_limiter)
        enforce = rate_limit.enforce_rate_limit("evaluations")

        with patch.object(rate_limit, "record_rate_limit_exceeded") as alert:
            enforce(_auth_context())

        alert.assert_not_called()


class TestRecordRateLimitExceeded:
    def test_emits_warning_with_org_and_project_tags(self) -> None:
        client = MagicMock()
        client.is_active.return_value = True
        scope = MagicMock()
        scope_cm = MagicMock()
        scope_cm.__enter__.return_value = scope

        with (
            patch.object(telemetry.sentry_sdk, "get_client", return_value=client),
            patch.object(telemetry.sentry_sdk, "new_scope", return_value=scope_cm),
            patch.object(telemetry.sentry_sdk, "capture_message") as capture,
        ):
            telemetry.record_rate_limit_exceeded(
                organization_id=7,
                project_id=70,
                category="evaluations",
                limit_per_minute=3,
            )

        capture.assert_called_once()
        assert capture.call_args.kwargs["level"] == "warning"
        scope.set_tag.assert_any_call("alert.type", "rate_limit_exceeded")
        scope.set_tag.assert_any_call("tenant.organization_id", 7)
        scope.set_tag.assert_any_call("tenant.project_id", 70)
        scope.set_tag.assert_any_call("route_category", "evaluations")
        scope.set_extra.assert_any_call("limit_per_minute", 3)

    def test_noop_when_sentry_inactive(self) -> None:
        client = MagicMock()
        client.is_active.return_value = False

        with (
            patch.object(telemetry.sentry_sdk, "get_client", return_value=client),
            patch.object(telemetry.sentry_sdk, "capture_message") as capture,
        ):
            telemetry.record_rate_limit_exceeded(
                organization_id=7,
                project_id=None,
                category="evaluations",
                limit_per_minute=3,
            )

        capture.assert_not_called()

    def test_sentry_failure_never_propagates(self) -> None:
        client = MagicMock()
        client.is_active.side_effect = RuntimeError("sentry exploded")

        with patch.object(telemetry.sentry_sdk, "get_client", return_value=client):
            telemetry.record_rate_limit_exceeded(
                organization_id=7,
                project_id=70,
                category="evaluations",
                limit_per_minute=3,
            )


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

    def test_v2_improve_prompt_returns_429_after_its_own_limit(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        for _ in range(settings.PROMPT_IMPROVEMENT_RATE_LIMIT_PER_MINUTE):
            resp = client.post(
                V2_IMPROVE_PROMPT_URL, headers=user_api_key_header, json={}
            )
            assert resp.status_code == 422

        resp = client.post(V2_IMPROVE_PROMPT_URL, headers=user_api_key_header, json={})

        assert resp.status_code == 429
        assert GENERIC_DETAIL.match(resp.json()["error"])

    def test_evaluation_and_improve_prompt_use_separate_buckets(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE):
            client.post(V2_EVALUATIONS_URL, headers=user_api_key_header, json={})

        resp = client.post(V2_IMPROVE_PROMPT_URL, headers=user_api_key_header, json={})

        assert resp.status_code == 422

    def test_v2_iteration_is_not_rate_limited(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE + 1):
            resp = client.post(V2_ITERATIONS_URL, headers=user_api_key_header, json={})
            assert resp.status_code == 422

    def test_v1_evaluation_is_not_rate_limited(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        url = f"{settings.API_V1_STR}/evaluations"
        for _ in range(settings.EVAL_RATE_LIMIT_PER_MINUTE + 1):
            resp = client.post(url, headers=user_api_key_header, json={})
            assert resp.status_code == 422
