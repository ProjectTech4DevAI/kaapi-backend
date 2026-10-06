import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from opentelemetry.trace import SpanKind, StatusCode, format_span_id
from opentelemetry.util.http import ExcludeList
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor
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


class TestSetupTelemetryInstrumentors:
    def test_enabled_calls_instrument_on_both(self) -> None:
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

    def test_redis_failure_does_not_propagate_and_botocore_still_instruments(
        self,
    ) -> None:
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
            telemetry.setup_telemetry()

        botocore.return_value.instrument.assert_called_once()

    def test_noop_when_otel_disabled(self) -> None:
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
    def test_uses_sentry_release_when_set(self) -> None:
        with patch.object(telemetry.settings, "SENTRY_RELEASE", "kaapi-backend@9.9.9"):
            assert telemetry.resolve_sentry_release() == "kaapi-backend@9.9.9"

    def test_falls_back_to_service_and_version(self) -> None:
        with (
            patch.object(telemetry.settings, "SENTRY_RELEASE", None),
            patch.object(telemetry.settings, "BACKEND_SERVICE_NAME", "kaapi-backend"),
            patch.object(telemetry.settings, "API_VERSION", "1.2.3"),
        ):
            assert telemetry.resolve_sentry_release() == "kaapi-backend@1.2.3"


class TestNotableSqlstates:
    def test_maps_known_postgres_codes(self) -> None:
        assert telemetry.NOTABLE_SQLSTATES["40P01"] == "deadlock_detected"
        assert telemetry.NOTABLE_SQLSTATES["57014"] == "query_canceled"
        assert telemetry.NOTABLE_SQLSTATES["40001"] == "serialization_failure"

    def test_unknown_code_absent(self) -> None:
        assert "99999" not in telemetry.NOTABLE_SQLSTATES


class TestRecordDbQueryFailed:
    def test_emits_count_with_operation_and_sqlstate(self) -> None:
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

    def test_omits_missing_attributes(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_query_failed()

        assert fake.metrics.count.call_args.kwargs["attributes"] == {}

    def test_noop_when_otel_disabled(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_query_failed(operation="SELECT", sqlstate="40P01")

        fake.metrics.count.assert_not_called()

    def test_noop_when_sentry_inactive(self) -> None:
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

    def test_known_code_sets_named_tag_and_span_attribute(self) -> None:
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

    def test_unknown_code_skips_error_name_tag(self) -> None:
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

    def test_none_sqlstate_is_noop(self) -> None:
        fake = _active_sentry()
        span = self._recording_span()
        with (
            patch.object(telemetry, "sentry_sdk", fake),
            patch.object(telemetry.trace, "get_current_span", return_value=span),
        ):
            telemetry._tag_db_error(None)

        fake.set_tag.assert_not_called()
        span.set_attribute.assert_not_called()

    def test_swallows_exceptions(self) -> None:
        fake = MagicMock()
        fake.get_client.side_effect = RuntimeError("sentry exploded")
        span = self._recording_span()
        with (
            patch.object(telemetry, "sentry_sdk", fake),
            patch.object(telemetry.trace, "get_current_span", return_value=span),
        ):
            telemetry._tag_db_error("40P01")


class TestSuppressDbInstrumentation:
    def test_scope_constant(self) -> None:
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

    def test_outside_cm_never_drops(self) -> None:
        assert telemetry._suppress_db_spans_var.get() is False
        assert telemetry._should_drop_db_span(self._sqlalchemy_span()) is False

    def test_inside_cm_drops_only_sqlalchemy_spans(self) -> None:
        with telemetry.suppress_db_instrumentation():
            assert telemetry._suppress_db_spans_var.get() is True
            assert telemetry._should_drop_db_span(self._sqlalchemy_span()) is True
            assert telemetry._should_drop_db_span(self._httpx_span()) is False

    def test_span_without_scope_not_dropped(self) -> None:
        with telemetry.suppress_db_instrumentation():
            assert telemetry._should_drop_db_span(SimpleNamespace()) is False

    def test_contextvar_resets_after_exit(self) -> None:
        with telemetry.suppress_db_instrumentation():
            assert telemetry._suppress_db_spans_var.get() is True
        assert telemetry._suppress_db_spans_var.get() is False

    def test_contextvar_resets_even_when_body_raises(self) -> None:
        try:
            with telemetry.suppress_db_instrumentation():
                raise ValueError("boom")
        except ValueError:
            pass
        assert telemetry._suppress_db_spans_var.get() is False


class TestInstrumentDbEngine:
    def _engine(self):
        return create_engine("sqlite://", poolclass=QueuePool)

    def test_successful_query_emits_pool_stats(self) -> None:
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

    def test_failing_query_fires_error_hook(self) -> None:
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

    def test_second_call_is_idempotent(self) -> None:
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
            instrumentor.return_value.instrument.assert_called_once()

    def test_noop_when_otel_disabled(self) -> None:
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

    def test_rowcount_attribute_set_on_query_span(self) -> None:
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
            def _attach_span(
                _conn, _cursor, _statement, _parameters, context, _executemany
            ):
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
        assert 1 in [c.args[1] for c in rows_calls]
        assert all(isinstance(c.args[1], int) for c in rows_calls)


class TestShouldDropBareHttpTrace:
    def test_drops_root_http_span_with_no_children(self) -> None:
        assert telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=False,
        )

    def test_keeps_when_trace_has_children(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=True,
        )

    def test_keeps_error_trace_even_without_children(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.SERVER,
            status_code=StatusCode.ERROR,
            had_children=False,
        )

    def test_keeps_non_root_span(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=False,
            kind=SpanKind.SERVER,
            status_code=StatusCode.UNSET,
            had_children=False,
        )

    def test_keeps_non_server_root(self) -> None:
        assert not telemetry._should_drop_bare_http_trace(
            is_root=True,
            kind=SpanKind.INTERNAL,
            status_code=StatusCode.UNSET,
            had_children=False,
        )


class TestNoiseFilteringSpanProcessor:
    @staticmethod
    def _root_server_span(span_id: int, start_time_ns: int) -> MagicMock:
        span = MagicMock()
        span.parent = None
        span.kind = SpanKind.SERVER
        span.status.status_code = StatusCode.UNSET
        span.start_time = start_time_ns
        span.get_span_context.return_value = SimpleNamespace(
            trace_id=0xABC, span_id=span_id, is_valid=True
        )
        return span

    def test_dropped_root_span_is_removed_from_open_spans(self) -> None:
        processor = telemetry._NoiseFilteringSpanProcessor()
        start_time_ns = int(time.time() * 1e9)
        span = self._root_server_span(span_id=0x1, start_time_ns=start_time_ns)
        span_id = format_span_id(0x1)
        started_minute = int(start_time_ns / 1e9 / 60)
        processor.otel_span_map[span_id] = MagicMock()
        processor.open_spans[started_minute] = {span_id}

        with patch.object(SentrySpanProcessor, "on_end") as base_on_end:
            processor.on_end(span)

        base_on_end.assert_not_called()
        assert span_id not in processor.otel_span_map
        assert all(span_id not in bucket for bucket in processor.open_spans.values())


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

    def test_health_and_cron_paths_excluded_from_traces(self) -> None:
        el = self._excluded()
        assert el.url_disabled("/health")
        assert el.url_disabled("/api/v1/utils/health")
        assert el.url_disabled("/api/v1/cron/run_batches")

    def test_framework_doc_paths_excluded_from_traces(self) -> None:
        el = self._excluded()
        assert el.url_disabled("/docs")
        assert el.url_disabled("/redoc")
        assert el.url_disabled("/api/v1/openapi.json")
        assert el.url_disabled("/docs/oauth2-redirect")

    def test_real_endpoints_still_traced(self) -> None:
        el = self._excluded()
        assert not el.url_disabled("/api/v1/llm/generate")
        assert not el.url_disabled("/api/v1/cronies")
        assert not el.url_disabled("/api/v1/documents")

    def test_noop_when_otel_disabled(self) -> None:
        instrument = MagicMock()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry.FastAPIInstrumentor, "instrument_app", instrument),
        ):
            telemetry.instrument_app(self._app())

        instrument.assert_not_called()


class TestRecordDbSlowQuery:
    def test_emits_count_with_operation(self) -> None:
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

    def test_noop_when_otel_disabled(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", False),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_slow_query("SELECT")

        fake.metrics.count.assert_not_called()


class TestRecordDbConnectionEvent:
    def test_known_event_emits_named_metric(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_connection_event("invalidated")

        assert (
            fake.metrics.count.call_args.kwargs["name"] == "db.connection.invalidated"
        )

    def test_unknown_event_is_noop(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_connection_event("teleported")

        fake.metrics.count.assert_not_called()

    def test_noop_when_otel_disabled(self) -> None:
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

    def test_unknown_outcome_is_noop(self) -> None:
        fake = _active_sentry()
        with (
            patch.object(telemetry.settings, "OTEL_ENABLED", True),
            patch.object(telemetry, "sentry_sdk", fake),
        ):
            telemetry.record_db_transaction("savepoint")

        fake.metrics.count.assert_not_called()


class TestInstrumentDbEngineMetrics:
    def _engine(self):
        return create_engine("sqlite://", poolclass=QueuePool)

    def test_slow_query_counted_when_over_threshold(self) -> None:
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

    def test_fast_query_not_counted_when_under_threshold(self) -> None:
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

    def test_connection_open_event_recorded(self) -> None:
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

    def test_commit_and_rollback_recorded(self) -> None:
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
