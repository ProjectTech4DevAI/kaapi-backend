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
from fastapi import FastAPI
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.http import ExcludeList
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
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


class TestSetRequestLogContext:
    @pytest.fixture(autouse=True)
    def _clean_log_context(self):
        token = telemetry._log_context_var.set(None)
        yield
        telemetry._log_context_var.reset(token)

    def test_binds_all_ids_stringified_to_sentry_user(self):
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5, user_id=7)

        fake.set_user.assert_called_once_with(
            {"id": "7", "org_id": "3", "project_id": "5"}
        )

    def test_only_present_keys_included_in_sentry_user(self):
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(user_id=7)

        fake.set_user.assert_called_once_with({"id": "7"})

    def test_all_ids_none_does_not_call_set_user(self):
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context()

        fake.set_user.assert_not_called()

    def test_inactive_sentry_does_not_call_set_user(self):
        fake = _inactive_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            # Must not raise even with all ids present.
            telemetry.set_request_log_context(org_id=3, project_id=5, user_id=7)

        fake.set_user.assert_not_called()

    def test_org_and_project_still_set_as_tags(self):
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5, user_id=7)

        fake.set_tag.assert_any_call("tenant.org_id", "3")
        fake.set_tag.assert_any_call("tenant.project_id", "5")

    def test_only_org_and_project_land_in_log_context(self):
        fake = _active_sentry()
        with patch.object(telemetry, "sentry_sdk", fake):
            telemetry.set_request_log_context(org_id=3, project_id=5, user_id=7)

        # user_id is scope-only; it must never enter the request log context.
        assert telemetry._log_context_var.get() == {"org_id": "3", "project_id": "5"}


class TestSetupTelemetryInstrumentors:
    """Assert Redis/Botocore auto-instrumentation wiring without mutating the
    global OTel provider: TracerProvider and set_tracer_provider are stubbed so
    setup_telemetry runs to the instrumentor calls but registers nothing real.
    """

    def test_enabled_calls_instrument_on_both(self):
        redis, botocore = MagicMock(), MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry.settings, "SENTRY_DSN", None),
            patch.object(telemetry, "TracerProvider", MagicMock()),
            patch.object(telemetry.trace, "set_tracer_provider", MagicMock()),
            patch.object(telemetry, "LoggingInstrumentor", MagicMock()),
            patch.object(telemetry, "HTTPXClientInstrumentor", MagicMock()),
            patch.object(telemetry, "RequestsInstrumentor", MagicMock()),
            patch(
                "opentelemetry.instrumentation.celery.CeleryInstrumentor", MagicMock()
            ),
            patch("opentelemetry.instrumentation.redis.RedisInstrumentor", redis),
            patch(
                "opentelemetry.instrumentation.botocore.BotocoreInstrumentor", botocore
            ),
        ):
            telemetry.setup_telemetry()

        redis.return_value.instrument.assert_called_once()
        botocore.return_value.instrument.assert_called_once()

    def test_redis_failure_does_not_propagate_and_botocore_still_instruments(self):
        redis = MagicMock()
        redis.return_value.instrument.side_effect = RuntimeError("boom")
        botocore = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry.settings, "SENTRY_DSN", None),
            patch.object(telemetry, "TracerProvider", MagicMock()),
            patch.object(telemetry.trace, "set_tracer_provider", MagicMock()),
            patch.object(telemetry, "LoggingInstrumentor", MagicMock()),
            patch.object(telemetry, "HTTPXClientInstrumentor", MagicMock()),
            patch.object(telemetry, "RequestsInstrumentor", MagicMock()),
            patch(
                "opentelemetry.instrumentation.celery.CeleryInstrumentor", MagicMock()
            ),
            patch("opentelemetry.instrumentation.redis.RedisInstrumentor", redis),
            patch(
                "opentelemetry.instrumentation.botocore.BotocoreInstrumentor", botocore
            ),
        ):
            # The defensive try/except must swallow the Redis failure.
            telemetry.setup_telemetry()

        botocore.return_value.instrument.assert_called_once()

    def test_noop_when_otel_disabled(self):
        redis, botocore = MagicMock(), MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch("opentelemetry.instrumentation.redis.RedisInstrumentor", redis),
            patch(
                "opentelemetry.instrumentation.botocore.BotocoreInstrumentor", botocore
            ),
        ):
            telemetry.setup_telemetry()

        redis.assert_not_called()
        botocore.assert_not_called()


class TestResolveSentryRelease:
    def test_uses_sentry_release_when_set(self):
        with patch.object(telemetry.settings, "SENTRY_RELEASE", "kaapi-backend@9.9.9"):
            assert telemetry.resolve_sentry_release() == "kaapi-backend@9.9.9"

    def test_falls_back_to_service_and_version(self):
        with (
            patch.object(telemetry.settings, "SENTRY_RELEASE", None),
            patch.object(telemetry.settings, "BACKEND_SERVICE_NAME", "kaapi-backend"),
            patch.object(telemetry.settings, "API_VERSION", "1.2.3"),
        ):
            assert telemetry.resolve_sentry_release() == "kaapi-backend@1.2.3"


class TestProfilingConstants:
    def test_continuous_profiling_enabled(self):
        assert telemetry.SENTRY_PROFILE_SESSION_SAMPLE_RATE == 1.0
        assert telemetry.SENTRY_PROFILE_LIFECYCLE == "trace"


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
                with pytest.raises(OperationalError):
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

    def test_rowcount_attribute_set_on_query_span(self):
        from sqlalchemy import event

        engine = self._engine()
        span = MagicMock()
        span.is_recording.return_value = True
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
        ):
            telemetry.instrument_db_engine(engine)

            # The stubbed instrumentor never populates context._otel_span, so stand in
            # for it: the rowcount listener reads the span off the execution context.
            @event.listens_for(engine, "before_cursor_execute")
            def _attach_span(conn, cursor, statement, parameters, context, executemany):
                context._otel_span = span

            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE t (x)"))
                conn.execute(text("INSERT INTO t VALUES (1)"))
                conn.commit()

        rows_calls = [
            c
            for c in span.set_attribute.call_args_list
            if c.args[0] == telemetry.DB_ROWS_ATTRIBUTE
        ]
        assert rows_calls
        # The INSERT affected one row; the count is an int, never row data.
        assert 1 in [c.args[1] for c in rows_calls]
        assert all(isinstance(c.args[1], int) for c in rows_calls)


class TestShouldDropBareHttpTrace:
    def test_drops_root_http_span_with_no_children(self):
        assert telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=False,
        )

    def test_keeps_when_trace_has_children(self):
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=True,
        )

    def test_keeps_error_trace_even_without_children(self):
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.ERROR,
            had_children=False,
        )

    def test_keeps_non_root_span(self):
        assert not telemetry._should_drop_bare_http_trace(
            is_root=False,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=False,
        )

    def test_keeps_non_server_root(self):
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.INTERNAL,
            status_code=StatusCode.UNSET,
            had_children=False,
        )


class TestInstrumentApp:
    def _app(self) -> FastAPI:
        return FastAPI(openapi_url="/api/v1/openapi.json")

    def _excluded(self) -> ExcludeList:
        instrument = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry.FastAPIInstrumentor, "instrument_app", instrument),
        ):
            telemetry.instrument_app(self._app())

        excluded_urls = instrument.call_args.kwargs["excluded_urls"]
        return ExcludeList(excluded_urls.split(","))

    def test_health_and_cron_paths_excluded_from_traces(self):
        el = self._excluded()
        assert el.url_disabled("/health")
        assert el.url_disabled("/api/v1/utils/health")
        assert el.url_disabled("/api/v1/cron/run_batches")

    def test_framework_doc_paths_excluded_from_traces(self):
        el = self._excluded()
        assert el.url_disabled("/docs")
        assert el.url_disabled("/redoc")
        assert el.url_disabled("/api/v1/openapi.json")
        assert el.url_disabled("/docs/oauth2-redirect")

    def test_real_endpoints_still_traced(self):
        el = self._excluded()
        assert not el.url_disabled("/api/v1/llm/generate")
        assert not el.url_disabled("/api/v1/cronies")
        assert not el.url_disabled("/api/v1/documents")

    def test_noop_when_otel_disabled(self):
        instrument = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry.FastAPIInstrumentor, "instrument_app", instrument),
        ):
            telemetry.instrument_app(self._app())

        instrument.assert_not_called()


class TestRecordDbSlowQuery:
    def test_emits_count_with_operation(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_slow_query("SELECT")

        fake.metrics.count.assert_called_once()
        kwargs = fake.metrics.count.call_args.kwargs
        assert kwargs["name"] == "db.query.slow"
        assert kwargs["value"] == 1
        assert kwargs["attributes"]["db.operation"] == "SELECT"

    def test_noop_when_otel_disabled(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_slow_query("SELECT")

        fake.metrics.count.assert_not_called()


class TestRecordDbConnectionEvent:
    def test_known_event_emits_named_metric(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_connection_event("invalidated")

        assert (
            fake.metrics.count.call_args.kwargs["name"] == "db.connection.invalidated"
        )

    def test_unknown_event_is_noop(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_connection_event("teleported")

        fake.metrics.count.assert_not_called()

    def test_noop_when_otel_disabled(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_connection_event("opened")

        fake.metrics.count.assert_not_called()


class TestRecordDbTransaction:
    @pytest.mark.parametrize(
        ("outcome", "metric"),
        [
            ("commit", "db.transaction.commit"),
            ("rollback", "db.transaction.rollback"),
        ],
    )
    def test_known_outcome_emits_named_metric(self, outcome, metric):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_transaction(outcome)

        assert fake.metrics.count.call_args.kwargs["name"] == metric

    def test_unknown_outcome_is_noop(self):
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_transaction("savepoint")

        fake.metrics.count.assert_not_called()


class TestInstrumentDbEngineMetrics:
    """Drive the new slow-query/connection/transaction hooks with a real SQLite engine."""

    def _engine(self):
        return create_engine("sqlite://", poolclass=QueuePool)

    def test_slow_query_counted_when_over_threshold(self):
        engine = self._engine()
        slow = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "DB_SLOW_QUERY_MS", 0),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
            patch.object(telemetry, "record_db_slow_query", slow),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        slow.assert_called()
        assert "SELECT" in [c.args[0] for c in slow.call_args_list]

    def test_fast_query_not_counted_when_under_threshold(self):
        engine = self._engine()
        slow = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "DB_SLOW_QUERY_MS", 10_000),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
            patch.object(telemetry, "record_db_slow_query", slow),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        slow.assert_not_called()

    def test_connection_open_event_recorded(self):
        engine = self._engine()
        conn_event = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
            patch.object(telemetry, "record_db_connection_event", conn_event),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        assert "opened" in [c.args[0] for c in conn_event.call_args_list]

    def test_commit_and_rollback_recorded(self):
        engine = self._engine()
        txn = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch("opentelemetry.instrumentation.sqlalchemy.SQLAlchemyInstrumentor"),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
            patch.object(telemetry, "record_db_transaction", txn),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE t (x)"))
                conn.commit()
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO t VALUES (1)"))
                conn.rollback()

        outcomes = [c.args[0] for c in txn.call_args_list]
        assert "commit" in outcomes
        assert "rollback" in outcomes
