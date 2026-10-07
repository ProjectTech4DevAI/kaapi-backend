import logging
from unittest.mock import MagicMock, patch

import pytest

from app.core.telemetry import context as telemetry


def _active_sentry() -> MagicMock:
    """A sentry_sdk stand-in whose client reports active."""
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = True
    return fake


def _inactive_sentry() -> MagicMock:
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = False
    return fake


class TestSetRequestLogContext:
    @pytest.fixture(autouse=True)
    def _clean_log_context(self):
        token = telemetry._log_context_var.set(None)
        yield
        telemetry._log_context_var.reset(token)

    def test_org_and_project_land_in_log_context(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5)

        assert telemetry._log_context_var.get() == {"org_id": "3", "project_id": "5"}

    def test_org_and_project_set_as_tags(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5)

        fake.set_tag.assert_any_call("kaapi.organization_id", "3")
        fake.set_tag.assert_any_call("kaapi.project_id", "5")

    def test_never_binds_a_sentry_user(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5)

        fake.set_user.assert_not_called()

    def test_inactive_sentry_still_sets_log_context(self) -> None:
        fake = _inactive_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5)

        assert telemetry._log_context_var.get() == {"org_id": "3", "project_id": "5"}
        fake.set_tag.assert_not_called()


class TestBindSentryUser:
    @pytest.fixture(autouse=True)
    def _clean_log_context(self):
        token = telemetry._log_context_var.set(None)
        yield
        telemetry._log_context_var.reset(token)

    def test_binds_all_ids_stringified_to_sentry_user(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.bind_sentry_user(user_id=7, org_id=3, project_id=5)

        fake.set_user.assert_called_once_with(
            {"id": "7", "org_id": "3", "project_id": "5"}
        )

    def test_only_present_keys_included_in_sentry_user(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.bind_sentry_user(user_id=7)

        fake.set_user.assert_called_once_with({"id": "7"})

    def test_all_ids_none_does_not_call_set_user(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.bind_sentry_user()

        fake.set_user.assert_not_called()

    def test_inactive_sentry_does_not_call_set_user(self) -> None:
        fake = _inactive_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.bind_sentry_user(user_id=7, org_id=3, project_id=5)

        fake.set_user.assert_not_called()

    def test_ids_never_enter_the_log_context(self) -> None:
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.bind_sentry_user(user_id=7, org_id=3, project_id=5)

        assert telemetry._log_context_var.get() is None


class TestLogContext:
    @pytest.fixture(autouse=True)
    def _clean_log_context(self):
        token = telemetry._log_context_var.set(None)
        yield
        telemetry._log_context_var.reset(token)

    def test_sets_tag_system_and_fields_then_resets(self) -> None:
        with telemetry.log_context(tag="llm-call", job_id="j1", skipped=None):
            assert telemetry._log_context_var.get() == {
                "tag": "llm-call",
                "system": "llm-call",
                "job_id": "j1",
            }
        assert not telemetry._log_context_var.get()

    def test_nested_context_extends_without_overriding_system(self) -> None:
        with telemetry.log_context(tag="collection"):
            with telemetry.log_context(tag="inner", step=2):
                payload = telemetry._log_context_var.get() or {}
                assert payload["tag"] == "inner"
                assert payload["system"] == "collection"
                assert payload["step"] == "2"


class TestLogContextFilter:
    @pytest.fixture(autouse=True)
    def _clean_log_context(self):
        token = telemetry._log_context_var.set(None)
        yield
        telemetry._log_context_var.reset(token)

    def _record(self, name: str) -> logging.LogRecord:
        return logging.LogRecord(name, logging.INFO, __file__, 1, "msg", None, None)

    def test_copies_context_fields_onto_record(self) -> None:
        record = self._record("app.anything")
        with telemetry.log_context(tag="custom", job_id="j1"):
            assert telemetry.LogContextFilter().filter(record)
        assert record.tag == "custom"
        assert record.job_id == "j1"

    def test_infers_llm_call_tag_from_logger_name(self) -> None:
        record = self._record("app.services.llm.jobs")
        telemetry.LogContextFilter().filter(record)
        assert record.tag == "llm-call"
        assert record.system == "llm-call"

    def test_infers_collection_tag_from_logger_name(self) -> None:
        record = self._record("app.crud.collection")
        telemetry.LogContextFilter().filter(record)
        assert record.tag == "collection"
        assert record.system == "collection"

    def test_stamps_lifecycle_from_recording_span(self) -> None:
        record = self._record("app.other")
        span = MagicMock()
        span.is_recording.return_value = True
        span.name = "run/app.task"
        with patch.object(telemetry.trace, "get_current_span", return_value=span):
            telemetry.LogContextFilter().filter(record)
        assert record.lifecycle == "run/app.task"
