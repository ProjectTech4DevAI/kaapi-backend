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

        fake.set_tag.assert_any_call("tenant.org_id", "3")
        fake.set_tag.assert_any_call("tenant.project_id", "5")

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
