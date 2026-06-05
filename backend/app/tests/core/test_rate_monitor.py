"""Tests for rate_monitor.py and the record_rate_threshold telemetry helper.
All Redis and Sentry calls are mocked; no real Redis or Sentry connection is used.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import redis

from app.core import rate_monitor, telemetry


def _auth_context(org_id: int | None = 1, org_name: str = "Acme"):
    """Build a minimal stand-in for AuthContext.

    monitor_rate's checker only reads auth_context.organization.id and .name,
    so a SimpleNamespace is enough — no DB or real models required.
    """
    org = None if org_id is None else SimpleNamespace(id=org_id, name=org_name)
    return SimpleNamespace(organization=org)


# ---------------------------------------------------------------------------
# increment_and_get_count
# ---------------------------------------------------------------------------


class TestIncrementAndGetCount:
    def test_returns_count_and_sets_expiry(self):
        """Pipeline runs INCR + EXPIRE and returns the incremented value."""
        pipe = MagicMock()
        pipe.execute.return_value = [5, True]  # [incr_result, expire_result]
        fake_redis = MagicMock()
        fake_redis.pipeline.return_value = pipe

        with patch.object(rate_monitor, "_redis_client", fake_redis):
            count = rate_monitor.increment_and_get_count("some-key")

        assert count == 5
        pipe.incr.assert_called_once_with("some-key")
        pipe.expire.assert_called_once_with(
            "some-key", rate_monitor._EXPIRATION_SECONDS
        )

    def test_returns_none_on_redis_error(self):
        """Any Redis failure is caught and returns None rather than raising."""
        fake_redis = MagicMock()
        fake_redis.pipeline.side_effect = redis.RedisError("boom")

        with patch.object(rate_monitor, "_redis_client", fake_redis):
            count = rate_monitor.increment_and_get_count("some-key")

        assert count is None


# ---------------------------------------------------------------------------
# monitor_rate / _checker
# ---------------------------------------------------------------------------


class TestMonitorRate:
    def test_skips_when_no_organization(self):
        """No org on the request → nothing counted, no Redis call."""
        checker = rate_monitor.monitor_rate("llm_call")

        with patch.object(rate_monitor, "increment_and_get_count") as inc:
            checker(_auth_context(org_id=None))

        inc.assert_not_called()

    def test_skips_when_no_threshold_for_category(self):
        """Unknown category has no threshold → return early, no alert."""
        checker = rate_monitor.monitor_rate("unknown")  # type: ignore[arg-type]

        with (
            patch.object(rate_monitor, "increment_and_get_count") as inc,
            patch.object(rate_monitor, "record_rate_threshold") as record,
        ):
            checker(_auth_context())

        inc.assert_not_called()
        record.assert_not_called()

    def test_no_alert_when_under_threshold(self):
        """Count at or below the threshold does not alert."""
        checker = rate_monitor.monitor_rate("collections")
        threshold = rate_monitor.THRESHOLDS["collections"]

        with (
            patch.object(
                rate_monitor, "increment_and_get_count", return_value=threshold
            ),
            patch.object(rate_monitor, "record_rate_threshold") as record,
        ):
            checker(_auth_context())

        record.assert_not_called()

    def test_alerts_when_over_threshold(self):
        """Count above the threshold records a Sentry alert with org details."""
        checker = rate_monitor.monitor_rate("llm_call")
        threshold = rate_monitor.THRESHOLDS["llm_call"]
        over = threshold + 1

        with (
            patch.object(rate_monitor, "increment_and_get_count", return_value=over),
            patch.object(rate_monitor, "record_rate_threshold") as record,
        ):
            checker(_auth_context(org_id=616, org_name="Acme"))

        record.assert_called_once_with(
            org_id=616,
            org_name="Acme",
            category="llm_call",
            request_count=over,
            threshold=threshold,
        )

    def test_no_alert_when_count_is_none(self):
        """increment returning None (Redis down) is treated as no breach."""
        checker = rate_monitor.monitor_rate("llm_call")

        with (
            patch.object(rate_monitor, "increment_and_get_count", return_value=None),
            patch.object(rate_monitor, "record_rate_threshold") as record,
        ):
            checker(_auth_context())

        record.assert_not_called()

    def test_redis_error_is_swallowed(self):
        """A RedisError from increment must not propagate out of the checker."""
        checker = rate_monitor.monitor_rate("llm_call")

        with patch.object(
            rate_monitor,
            "increment_and_get_count",
            side_effect=redis.RedisError("down"),
        ):
            # Should not raise.
            checker(_auth_context())


# ---------------------------------------------------------------------------
# telemetry.record_rate_threshold
# ---------------------------------------------------------------------------


class TestRecordRateThreshold:
    def test_emits_warning_message_with_tags(self):
        """When Sentry is active, a warning message is captured with tags/extras."""
        client = MagicMock()
        client.is_active.return_value = True
        scope = MagicMock()
        scope_cm = MagicMock()
        scope_cm.__enter__.return_value = scope

        with (
            patch.object(telemetry.sentry_sdk, "get_client", return_value=client),
            patch.object(telemetry.sentry_sdk, "push_scope", return_value=scope_cm),
            patch.object(telemetry.sentry_sdk, "capture_message") as capture,
        ):
            telemetry.record_rate_threshold(
                org_id=616,
                org_name="Acme",
                category="llm_call",
                request_count=16,
                threshold=15,
            )

        capture.assert_called_once()
        assert capture.call_args.kwargs["level"] == "warning"
        scope.set_tag.assert_any_call("alert.type", "threshold_rate_monitor")
        scope.set_tag.assert_any_call("tenant.org_id", 616)
        scope.set_extra.assert_any_call("request_count", 16)
        scope.set_extra.assert_any_call("threshold", 15)

    def test_noop_when_sentry_inactive(self):
        """No Sentry client → nothing is captured."""
        client = MagicMock()
        client.is_active.return_value = False

        with (
            patch.object(telemetry.sentry_sdk, "get_client", return_value=client),
            patch.object(telemetry.sentry_sdk, "capture_message") as capture,
        ):
            telemetry.record_rate_threshold(
                org_id=1,
                org_name="Acme",
                category="llm_call",
                request_count=16,
                threshold=15,
            )

        capture.assert_not_called()

    def test_swallows_exceptions(self):
        """An error inside Sentry emission must never propagate."""
        client = MagicMock()
        client.is_active.side_effect = RuntimeError("sentry exploded")

        with patch.object(telemetry.sentry_sdk, "get_client", return_value=client):
            # Should not raise.
            telemetry.record_rate_threshold(
                org_id=1,
                org_name="Acme",
                category="llm_call",
                request_count=16,
                threshold=15,
            )
