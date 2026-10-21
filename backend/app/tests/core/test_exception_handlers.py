import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import sentry_sdk
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.types import Event, Hint
from sqlalchemy.exc import (
    DataError,
    OperationalError,
    ProgrammingError,
)

from app.core.config import settings
from app.core.exception_handlers import (
    DB_ERROR_DETAIL,
    GENERIC_ERROR_DETAIL,
    _sanitize_validation_errors,
    register_exception_handlers,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.utils import make_integrity_error


class TestSanitizeValidationErrors:
    """Unit tests for _sanitize_validation_errors."""

    def test_non_union_errors_pass_through(self) -> None:
        errors = [
            {"type": "missing", "loc": ("body", "name"), "msg": "Field required"},
        ]
        assert _sanitize_validation_errors(errors) == errors

    def test_picks_branch_with_fewer_literal_errors(self) -> None:
        errors = [
            {
                "type": "literal_error",
                "loc": ("body", "c", "NativeConfig", "provider"),
                "msg": "bad",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "NativeConfig", "params"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "KaapiConfig", "type"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "KaapiConfig", "params"),
                "msg": "Field required",
            },
        ]
        result = _sanitize_validation_errors(errors)
        assert len(result) == 2
        for err in result:
            assert "NativeConfig" not in err["loc"]

    def test_tied_branches_keep_both_and_dedup(self) -> None:
        """When two branches have the same literal_error count, both are kept but duplicates removed."""
        errors = [
            {
                "type": "missing",
                "loc": ("body", "c", "BranchA", "x"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "BranchB", "x"),
                "msg": "Field required",
            },
        ]
        result = _sanitize_validation_errors(errors)
        assert len(result) == 1
        assert result[0]["loc"] == ("body", "c", "x")

    def test_strips_branch_identifiers_from_loc(self) -> None:
        errors = [
            {
                "type": "missing",
                "loc": (
                    "body",
                    "cfg",
                    "completion",
                    "function-after[validate_params(), Foo]",
                    "params",
                ),
                "msg": "Field required",
            }
        ]
        result = _sanitize_validation_errors(errors)
        assert result[0]["loc"] == ("body", "cfg", "completion", "params")

    def test_non_union_preserved_with_union(self) -> None:
        errors = [
            {"type": "missing", "loc": ("body", "name"), "msg": "Field required"},
            {
                "type": "literal_error",
                "loc": ("body", "c", "NativeConfig", "p"),
                "msg": "bad",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "KaapiConfig", "t"),
                "msg": "Field required",
            },
        ]
        result = _sanitize_validation_errors(errors)
        assert len(result) == 2
        locs = [r["loc"] for r in result]
        assert ("body", "name") in locs

    def test_empty_list(self) -> None:
        assert _sanitize_validation_errors([]) == []

    def test_fallback_on_malformed_input(self) -> None:
        malformed = [None, 42]  # type: ignore[list-item]
        result = _sanitize_validation_errors(malformed)
        assert result == malformed


class TestSentryCaptureRouting:
    """Only the generic 500 handler reports to Sentry; 4xx handlers stay quiet."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()
        register_exception_handlers(app)

        class Body(BaseModel):
            x: int

        @app.get("/boom")
        def boom() -> None:
            raise ValueError("boom")

        @app.get("/http-error")
        def http_error() -> None:
            raise HTTPException(status_code=404, detail="nope")

        @app.post("/validate")
        def validate(body: Body) -> dict[str, bool]:
            return {"ok": True}

        return TestClient(app, raise_server_exceptions=False)

    def test_generic_handler_reports_unhandled_exception(
        self, client: TestClient
    ) -> None:
        with patch("app.core.exception_handlers.logger.error") as log_error:
            response = client.get("/boom")

        assert response.status_code == 500
        log_error.assert_called_once()

    def test_http_exception_handler_does_not_capture(self, client: TestClient) -> None:
        with patch(
            "app.core.exception_handlers.sentry_sdk.capture_exception"
        ) as capture:
            response = client.get("/http-error")

        assert response.status_code == 404
        capture.assert_not_called()

    def test_validation_error_handler_does_not_capture(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.core.exception_handlers.sentry_sdk.capture_exception"
        ) as capture:
            response = client.post("/validate", json={})

        assert response.status_code == 422
        capture.assert_not_called()


class TestValidationErrorResponse:
    """Integration: structured errors via configs endpoint."""

    def test_structured_error_format(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        response = client.post(
            f"{settings.API_V1_STR}/configs",
            headers={"X-API-KEY": user_api_key.key},
            json={},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "Validation failed"
        assert isinstance(body["errors"], list)
        assert all("field" in e and "message" in e for e in body["errors"])

    def test_union_noise_filtered(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        response = client.post(
            f"{settings.API_V1_STR}/configs",
            headers={"X-API-KEY": user_api_key.key},
            json={
                "name": "test-config",
                "config_blob": {"completion": {"provider": "openai"}},
            },
        )
        assert response.status_code == 422
        for error in response.json()["errors"]:
            assert "openai-native" not in error["message"]
            assert "NativeCompletionConfig" not in error["field"]


def _app_raising(exc: Exception) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return TestClient(app, raise_server_exceptions=False)


class TestGenericErrorHandler:
    def test_does_not_leak_exception_text(self) -> None:
        response = _app_raising(ValueError("secret arn:aws:kms:key/abc")).get("/boom")

        assert response.status_code == 500
        assert response.json()["error"] == GENERIC_ERROR_DETAIL
        assert "arn:aws:kms" not in response.text


class TestDbErrorHandler:
    @pytest.mark.parametrize(
        ("exc", "status_code"),
        [
            (make_integrity_error("23505"), 409),
            (make_integrity_error("23503"), 422),
            (make_integrity_error("23502"), 500),
            (DataError("INSERT", {}, Exception("too long")), 422),
            (OperationalError("SELECT", {}, Exception("conn refused")), 503),
            (ProgrammingError("SELECT", {}, Exception("bad sql")), 500),
        ],
    )
    def test_maps_status_without_leaking_sql(
        self, exc: Exception, status_code: int
    ) -> None:
        response = _app_raising(exc).get("/boom")

        assert response.status_code == status_code
        assert response.json()["error"].startswith("Database error:")
        assert "INSERT" not in response.text
        assert "9876543210" not in response.text

    def test_unmapped_db_error_uses_default_detail(self) -> None:
        response = _app_raising(ProgrammingError("SELECT", {}, Exception("x"))).get(
            "/boom"
        )

        assert response.json()["error"] == DB_ERROR_DETAIL

    @pytest.mark.parametrize(
        ("constraint", "detail"),
        [
            ("uq_config_project_id_name_active", "A config with this name"),
            ("uq_credential_org_project_provider", "Credentials for this provider"),
            ("ix_user_email", "A user with this email"),
        ],
    )
    def test_known_unique_constraint_gets_specific_message(
        self, constraint: str, detail: str
    ) -> None:
        response = _app_raising(make_integrity_error("23505", constraint)).get("/boom")

        assert response.status_code == 409
        assert response.json()["error"].startswith(detail)

    def test_unknown_unique_constraint_gets_generic_conflict(self) -> None:
        response = _app_raising(make_integrity_error("23505", "uq_other")).get("/boom")

        assert response.status_code == 409
        assert response.json()["error"].startswith("Database error:")

    def test_client_side_db_error_logs_warning(self) -> None:
        with (
            patch("app.core.exception_handlers.logger.warning") as log_warning,
            patch("app.core.exception_handlers.logger.error") as log_error,
        ):
            _app_raising(make_integrity_error("23505")).get("/boom")

        log_warning.assert_called_once()
        log_error.assert_not_called()


class TestHttpExceptionHandler:
    def test_server_error_is_logged(self) -> None:
        with patch("app.core.exception_handlers.logger.error") as log_error:
            response = _app_raising(
                HTTPException(status_code=502, detail="upstream failed")
            ).get("/boom")

        assert response.status_code == 502
        log_error.assert_called_once()

    def test_tags_status_class_when_sentry_active(self) -> None:
        client = MagicMock()
        client.is_active.return_value = True
        with (
            patch(
                "app.core.exception_handlers.sentry_sdk.get_client",
                return_value=client,
            ),
            patch("app.core.exception_handlers.sentry_sdk.set_tag") as set_tag,
        ):
            _app_raising(HTTPException(status_code=404, detail="nope")).get("/boom")

        set_tag.assert_any_call("http.status_code", "404")
        set_tag.assert_any_call("error.class", "client")


@pytest.fixture
def sentry_events() -> Iterator[list[Event]]:
    """Real Sentry client whose before_send records events instead of sending them."""
    events: list[Event] = []

    def _record(event: Event, _hint: Hint) -> None:
        events.append(event)
        return None

    sentry_sdk.init(
        dsn="https://public@sentry.invalid/1",
        before_send=_record,
        default_integrations=False,
        auto_enabling_integrations=False,
        integrations=[
            LoggingIntegration(event_level=logging.ERROR),
            DedupeIntegration(),
        ],
    )
    yield events
    sentry_sdk.get_global_scope().set_client(None)


class TestSentryEvents:
    def test_server_http_exception_event_carries_status_tags(
        self, sentry_events: list[Event]
    ) -> None:
        _app_raising(HTTPException(status_code=502, detail="upstream")).get("/boom")

        assert len(sentry_events) == 1
        tags = sentry_events[0]["tags"]
        assert tags["http.status_code"] == "502"
        assert tags["error.class"] == "server"

    def test_generic_handler_reports_exactly_one_event(
        self, sentry_events: list[Event]
    ) -> None:
        _app_raising(ValueError("boom")).get("/boom")

        assert len(sentry_events) == 1

    def test_db_server_error_event_carries_db_tags(
        self, sentry_events: list[Event]
    ) -> None:
        _app_raising(make_integrity_error("23502", "llm_call_input_not_null")).get(
            "/boom"
        )

        assert len(sentry_events) == 1
        tags = sentry_events[0]["tags"]
        assert tags["http.status_code"] == "500"
        assert tags["db.constraint"] == "llm_call_input_not_null"

    def test_db_conflict_creates_no_event(self, sentry_events: list[Event]) -> None:
        response = _app_raising(make_integrity_error("23505")).get("/boom")

        assert response.status_code == 409
        assert sentry_events == []
