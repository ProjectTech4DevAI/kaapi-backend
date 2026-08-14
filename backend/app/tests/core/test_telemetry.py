"""Tests for the DB-observability helpers in telemetry.py.

Sentry and OTel emission are mocked; no real Sentry connection or OTel provider
is used. The instrument_db_engine tests DO use a real in-memory SQLite engine to
drive the SQLAlchemy event hooks, but stub out the span instrumentor and the
Sentry-backed emit helpers. Most emitters gate on settings.OTEL_ENABLED, so the
enabled cases patch it True.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from app.core import telemetry


def _active_sentry() -> MagicMock:
    """A sentry_sdk stand-in whose client reports active."""
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = True
    return fake


def _inactive_sentry() -> MagicMock:
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = False
    return fake


class TestNotableSqlstates:
    def test_maps_known_postgres_codes(self):
        assert telemetry.NOTABLE_SQLSTATES["40P01"] == "deadlock_detected"
        assert telemetry.NOTABLE_SQLSTATES["57014"] == "query_canceled"
        assert telemetry.NOTABLE_SQLSTATES["40001"] == "serialization_failure"

    def test_unknown_code_absent(self):
        assert "99999" not in telemetry.NOTABLE_SQLSTATES


class TestRecordDbQueryFailed:
    def test_emits_count_with_operation_and_sqlstate(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_query_failed(operation="SELECT", sqlstate="40P01")

        fake.metrics.count.assert_called_once()
        kwargs = fake.metrics.count.call_args.kwargs
        assert kwargs["name"] == "db.query.failed"
        assert kwargs["value"] == 1
        assert kwargs["attributes"]["db.operation"] == "SELECT"
        assert kwargs["attributes"]["db.sqlstate"] == "40P01"

    def test_omits_missing_attributes(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_query_failed()

        assert fake.metrics.count.call_args.kwargs["attributes"] == {}

    def test_noop_when_otel_disabled(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_query_failed(operation="SELECT", sqlstate="40P01")

        fake.metrics.count.assert_not_called()

    def test_noop_when_sentry_inactive(self):
        fake = _inactive_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_query_failed(operation="SELECT", sqlstate="40P01")

        fake.metrics.count.assert_not_called()


class TestTagDbError:
    def _recording_span(self) -> MagicMock:
        span = MagicMock()
        span.is_recording.return_value = True
        return span

    def test_known_code_sets_named_tag_and_span_attribute(self):
        fake = _active_sentry()
        span = self._recording_span()
        with (
            patch.object(telemetry, "sentry_sdk", fake),
            patch.object(telemetry.trace, "get_current_span", return_value=span),
        ):
            telemetry._tag_db_error("40P01")

        fake.set_tag.assert_any_call("db.system", "postgresql")
        fake.set_tag.assert_any_call("db.sqlstate", "40P01")
        fake.set_tag.assert_any_call("db.error.name", "deadlock_detected")
        span.set_attribute.assert_called_once_with("db.sqlstate", "40P01")

    def test_unknown_code_skips_error_name_tag(self):
        fake = _active_sentry()
        span = self._recording_span()
        with (
            patch.object(telemetry, "sentry_sdk", fake),
            patch.object(telemetry.trace, "get_current_span", return_value=span),
        ):
            telemetry._tag_db_error("99999")

        fake.set_tag.assert_any_call("db.system", "postgresql")
        fake.set_tag.assert_any_call("db.sqlstate", "99999")
        tag_names = [c.args[0] for c in fake.set_tag.call_args_list]
        assert "db.error.name" not in tag_names

    def test_none_sqlstate_is_noop(self):
        fake = _active_sentry()
        span = self._recording_span()
        with (
            patch.object(telemetry, "sentry_sdk", fake),
            patch.object(telemetry.trace, "get_current_span", return_value=span),
        ):
            telemetry._tag_db_error(None)

        fake.set_tag.assert_not_called()
        span.set_attribute.assert_not_called()

    def test_swallows_exceptions(self):
        fake = MagicMock()
        fake.get_client.side_effect = RuntimeError("sentry exploded")
        span = self._recording_span()
        with (
            patch.object(telemetry, "sentry_sdk", fake),
            patch.object(telemetry.trace, "get_current_span", return_value=span),
        ):
            # Must not raise.
            telemetry._tag_db_error("40P01")


class TestSuppressDbInstrumentation:
    def test_scope_constant(self):
        assert telemetry._SQLALCHEMY_SCOPE == "opentelemetry.instrumentation.sqlalchemy"

    def _sqlalchemy_span(self) -> SimpleNamespace:
        return SimpleNamespace(
            instrumentation_scope=SimpleNamespace(
                name="opentelemetry.instrumentation.sqlalchemy"
            )
        )

    def _httpx_span(self) -> SimpleNamespace:
        return SimpleNamespace(
            instrumentation_scope=SimpleNamespace(
                name="opentelemetry.instrumentation.httpx"
            )
        )

    def test_outside_cm_never_drops(self):
        assert telemetry._suppress_db_spans_var.get() is False
        assert telemetry._should_drop_db_span(self._sqlalchemy_span()) is False

    def test_inside_cm_drops_only_sqlalchemy_spans(self):
        with telemetry.suppress_db_instrumentation():
            assert telemetry._suppress_db_spans_var.get() is True
            assert telemetry._should_drop_db_span(self._sqlalchemy_span()) is True
            assert telemetry._should_drop_db_span(self._httpx_span()) is False

    def test_span_without_scope_not_dropped(self):
        with telemetry.suppress_db_instrumentation():
            assert telemetry._should_drop_db_span(SimpleNamespace()) is False

    def test_contextvar_resets_after_exit(self):
        with telemetry.suppress_db_instrumentation():
            assert telemetry._suppress_db_spans_var.get() is True
        assert telemetry._suppress_db_spans_var.get() is False

    def test_contextvar_resets_even_when_body_raises(self):
        try:
            with telemetry.suppress_db_instrumentation():
                raise ValueError("boom")
        except ValueError:
            pass
        assert telemetry._suppress_db_spans_var.get() is False


class TestInstrumentDbEngine:
    """Drive the SQLAlchemy event hooks with a real in-memory SQLite engine.

    The span instrumentor is stubbed (no OTel provider needed) and the
    Sentry-backed emit helpers are patched so we assert on the hooks alone.
    """

    def _engine(self):
        return create_engine("sqlite://", poolclass=QueuePool)

    def test_successful_query_emits_pool_stats(self):
        engine = self._engine()
        pool_stats = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", pool_stats),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        pool_stats.assert_called()
        kwargs = pool_stats.call_args.kwargs
        assert set(kwargs) == {"active", "idle", "total", "overflow"}

    def test_failing_query_fires_error_hook(self):
        engine = self._engine()
        query_failed = MagicMock()
        tag_error = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
            patch.object(telemetry, "record_db_query_failed", query_failed),
            patch.object(telemetry, "_tag_db_error", tag_error),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                with pytest.raises(Exception):
                    conn.execute(text("SELECT * FROM does_not_exist"))

        query_failed.assert_called_once()
        # SQLite driver errors carry no sqlstate, so the operation is SELECT and
        # sqlstate is None — the hook still runs end to end.
        assert query_failed.call_args.kwargs["operation"] == "SELECT"
        assert query_failed.call_args.kwargs["sqlstate"] is None
        tag_error.assert_called_once_with(None)

    def test_second_call_is_idempotent(self):
        engine = self._engine()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch(
                "opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"
            ) as instrumentor,
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
        ):
            telemetry.instrument_db_engine(engine)
            assert engine._kaapi_db_telemetry_instrumented is True
            instrumentor.return_value.instrument.assert_called_once()

            telemetry.instrument_db_engine(engine)
            # Guard short-circuits: the instrumentor is not invoked a second time.
            instrumentor.return_value.instrument.assert_called_once()

    def test_noop_when_otel_disabled(self):
        engine = self._engine()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch(
                "opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"
            ) as instrumentor,
        ):
            telemetry.instrument_db_engine(engine)

        instrumentor.assert_not_called()
        assert not getattr(engine, "_kaapi_db_telemetry_instrumented", False)
