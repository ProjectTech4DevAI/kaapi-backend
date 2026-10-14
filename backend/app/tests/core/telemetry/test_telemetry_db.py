from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool

from app.core.telemetry import db as telemetry
from app.core.telemetry import metrics


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
    def test_maps_known_postgres_codes(self) -> None:
        assert telemetry.NOTABLE_SQLSTATES["40P01"] == "deadlock_detected"
        assert telemetry.NOTABLE_SQLSTATES["57014"] == "query_canceled"
        assert telemetry.NOTABLE_SQLSTATES["40001"] == "serialization_failure"

    def test_unknown_code_absent(self) -> None:
        assert "99999" not in telemetry.NOTABLE_SQLSTATES


class TestRecordDbQueryFailed:
    def test_emits_count_with_operation_and_sqlstate(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_query_failed(operation="SELECT", sqlstate="40P01")

        fake.metrics.count.assert_called_once()
        kwargs = fake.metrics.count.call_args.kwargs
        assert kwargs["name"] == "db.query.failed"
        assert kwargs["value"] == 1
        assert kwargs["attributes"]["db.operation"] == "SELECT"
        assert kwargs["attributes"]["db.sqlstate"] == "40P01"

    def test_omits_missing_attributes(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_query_failed()

        assert fake.metrics.count.call_args.kwargs["attributes"] == {}

    def test_noop_when_sentry_inactive(self) -> None:
        fake = _inactive_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_query_failed(operation="SELECT", sqlstate="40P01")

        fake.metrics.count.assert_not_called()


class TestRecordDbSlowQuery:
    def test_emits_count_with_operation(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_slow_query("SELECT")

        fake.metrics.count.assert_called_once()
        kwargs = fake.metrics.count.call_args.kwargs
        assert kwargs["name"] == "db.query.slow"
        assert kwargs["value"] == 1
        assert kwargs["attributes"]["db.operation"] == "SELECT"


class TestRecordDbConnectionEvent:
    def test_known_event_emits_named_metric(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_connection_event("invalidated")

        assert (
            fake.metrics.count.call_args.kwargs["name"] == "db.connection.invalidated"
        )

    def test_unknown_event_is_noop(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_connection_event("teleported")

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
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_transaction(outcome)

        assert fake.metrics.count.call_args.kwargs["name"] == metric

    def test_unknown_outcome_is_noop(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_transaction("savepoint")

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


class TestInstrumentDbEngine:
    def _engine(self):
        return create_engine("sqlite://", poolclass=QueuePool)

    def test_successful_query_emits_pool_stats(self) -> None:
        engine = self._engine()
        pool_stats = MagicMock()
        with (
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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
            patch.object(telemetry, "SQLAlchemyInstrumentor") as instrumentor,
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
        ):
            telemetry.instrument_db_engine(engine)
            assert engine._kaapi_db_telemetry_instrumented is True
            instrumentor.return_value.instrument.assert_called_once()

            telemetry.instrument_db_engine(engine)
            instrumentor.return_value.instrument.assert_called_once()

    def test_rowcount_attribute_set_on_query_span(self) -> None:
        from sqlalchemy import event

        engine = self._engine()
        span = MagicMock()
        span.is_recording.return_value = True
        with (
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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


class TestInstrumentDbEngineMetrics:
    def _engine(self):
        return create_engine("sqlite://", poolclass=QueuePool)

    def test_slow_query_counted_when_over_threshold(self) -> None:
        engine = self._engine()
        slow = MagicMock()
        with (
            patch.object(telemetry, "DB_SLOW_QUERY_MS", 0),
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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
            patch.object(telemetry, "DB_SLOW_QUERY_MS", 10_000),
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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
            patch.object(telemetry, "SQLAlchemyInstrumentor"),
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


class TestRecordDbPoolStats:
    def test_emits_four_gauges(self) -> None:
        fake = _active_sentry()
        with (patch.object(metrics, "sentry_sdk", fake),):
            telemetry.record_db_pool_stats(active=1, idle=2, total=3, overflow=0)
        names = [c.kwargs["name"] for c in fake.metrics.gauge.call_args_list]
        assert names == [
            "db.pool.active",
            "db.pool.idle",
            "db.pool.total",
            "db.pool.overflow",
        ]


class TestInstrumentDbEngineEdgeCases:
    def test_non_queue_pool_skips_pool_metrics(self) -> None:
        engine = create_engine("sqlite://")
        pool_stats = MagicMock()
        with (
            patch.object(telemetry, "SQLAlchemyInstrumentor", MagicMock()),
            patch.object(telemetry, "record_db_pool_stats", pool_stats),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        pool_stats.assert_not_called()

    def test_close_and_invalidate_emit_connection_events(self) -> None:
        engine = create_engine("sqlite://", poolclass=QueuePool)
        events = MagicMock()
        with (
            patch.object(telemetry, "SQLAlchemyInstrumentor", MagicMock()),
            patch.object(telemetry, "record_db_pool_stats", MagicMock()),
            patch.object(telemetry, "record_db_connection_event", events),
        ):
            telemetry.instrument_db_engine(engine)
            with engine.connect() as conn:
                conn.invalidate()
            engine.dispose()
        emitted = {c.args[0] for c in events.call_args_list}
        assert {"opened", "invalidated", "closed"} <= emitted
