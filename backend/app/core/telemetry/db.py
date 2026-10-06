import logging
import time

import sentry_sdk
from opentelemetry import trace
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from sqlalchemy import ExceptionContext, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.interfaces import DBAPIConnection, DBAPICursor, ExecutionContext
from sqlalchemy.pool import ConnectionPoolEntry, Pool, PoolProxiedConnection, QueuePool

from app.core.config import settings
from app.core.telemetry.metrics import emit_sentry_metric

logger = logging.getLogger(__name__)

DB_SLOW_QUERY_MS: int = 500

DB_ROWS_ATTRIBUTE: str = "db.rows_affected"

_INSTRUMENTED_ATTR = "_kaapi_db_telemetry_instrumented"
_STARTED_AT_ATTR = "_kaapi_db_started_at"
_OPERATION_ATTR = "_kaapi_db_operation"

NOTABLE_SQLSTATES: dict[str, str] = {
    "40P01": "deadlock_detected",
    "57014": "query_canceled",  # statement timeout
    "40001": "serialization_failure",
    "55P03": "lock_not_available",  # lock timeout
    "08006": "connection_failure",
    "08003": "connection_does_not_exist",
    "53300": "too_many_connections",
}

_DB_CONNECTION_EVENT_METRICS: dict[str, str] = {
    "opened": "db.connection.opened",
    "closed": "db.connection.closed",
    "invalidated": "db.connection.invalidated",
}

_DB_TRANSACTION_METRICS: dict[str, str] = {
    "commit": "db.transaction.commit",
    "rollback": "db.transaction.rollback",
}


def record_db_query_failed(
    *,
    operation: str | None = None,
    sqlstate: str | None = None,
) -> None:
    """Emit a DB query-failure counter to Sentry. Per-query duration/throughput come from spans."""
    if not settings.OTEL_ENABLED:
        return

    attrs: dict[str, str | int | float] = {}
    if operation:
        attrs["db.operation"] = operation
    if sqlstate:
        attrs["db.sqlstate"] = sqlstate

    emit_sentry_metric("count", "db.query.failed", 1, attributes=attrs)


def record_db_slow_query(operation: str | None = None) -> None:
    """Emit a slow-query counter to Sentry (queries at/above DB_SLOW_QUERY_MS)."""
    if not settings.OTEL_ENABLED:
        return

    attrs: dict[str, str | int | float] = {}
    if operation:
        attrs["db.operation"] = operation

    emit_sentry_metric("count", "db.query.slow", 1, attributes=attrs)


def record_db_connection_event(event: str) -> None:
    """Emit a connection-lifecycle counter (opened/closed/invalidated) to Sentry."""
    if not settings.OTEL_ENABLED:
        return

    metric = _DB_CONNECTION_EVENT_METRICS.get(event)
    if metric is None:
        return
    emit_sentry_metric("count", metric, 1)


def record_db_transaction(outcome: str) -> None:
    """Emit a transaction-outcome counter (commit/rollback) to Sentry for ratio tracking."""
    if not settings.OTEL_ENABLED:
        return

    metric = _DB_TRANSACTION_METRICS.get(outcome)
    if metric is None:
        return
    emit_sentry_metric("count", metric, 1)


def _tag_db_error(sqlstate: str | None) -> None:
    """Tag the active Sentry scope + span with a DB error's Postgres SQLSTATE."""
    if not sqlstate:
        return
    try:
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("db.sqlstate", sqlstate)
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("db.system", "postgresql")
            sentry_sdk.set_tag("db.sqlstate", sqlstate)
            name = NOTABLE_SQLSTATES.get(sqlstate)
            if name:
                sentry_sdk.set_tag("db.error.name", name)
    except Exception:
        logger.debug("[_tag_db_error] Failed to tag DB error | sqlstate: %s", sqlstate)


def record_db_pool_stats(
    *,
    active: int,
    idle: int,
    total: int,
    overflow: int,
) -> None:
    """Emit SQLAlchemy pool stats as Sentry gauges."""
    if not settings.OTEL_ENABLED:
        return

    emit_sentry_metric("gauge", "db.pool.active", active)
    emit_sentry_metric("gauge", "db.pool.idle", idle)
    emit_sentry_metric("gauge", "db.pool.total", total)
    emit_sentry_metric("gauge", "db.pool.overflow", overflow)


def instrument_db_engine(engine: Engine) -> None:
    """Instrument a SQLAlchemy engine: query spans, slow-query/pool/connection/transaction metrics."""
    if not settings.OTEL_ENABLED:
        return
    if getattr(engine, _INSTRUMENTED_ATTR, False):
        return

    # ProxyTracer defers to the provider set later in setup_telemetry(); load order is safe.
    try:
        SQLAlchemyInstrumentor().instrument(engine=engine)
    except Exception:
        logger.exception(
            "[instrument_db_engine] Failed to load SQLAlchemy span instrumentation"
        )

    def _pool_snapshot(pool: Pool) -> tuple[int, int, int, int] | None:
        if not isinstance(pool, QueuePool):
            return None
        active = pool.checkedout()
        idle = pool.checkedin()
        overflow = pool.overflow()
        total = max(pool.size() + overflow, active + idle)
        return active, idle, total, overflow

    def _emit_pool_metrics(pool: Pool) -> None:
        snapshot = _pool_snapshot(pool)
        if not snapshot:
            return
        active, idle, total, overflow = snapshot
        record_db_pool_stats(active=active, idle=idle, total=total, overflow=overflow)

    @event.listens_for(engine, "before_cursor_execute")
    def _before_cursor_execute(
        conn: Connection,
        cursor: DBAPICursor,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        del cursor, parameters, executemany
        setattr(context, _STARTED_AT_ATTR, time.perf_counter())
        setattr(
            context,
            _OPERATION_ATTR,
            statement.split(None, 1)[0].upper() if statement else "UNKNOWN",
        )
        _emit_pool_metrics(conn.engine.pool)

    @event.listens_for(engine, "after_cursor_execute")
    def _after_cursor_execute(
        conn: Connection,
        cursor: DBAPICursor,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        del cursor, statement, parameters, executemany
        started_at = getattr(context, _STARTED_AT_ATTR, None)
        if started_at is not None:
            duration_ms = (time.perf_counter() - started_at) * 1000
            if duration_ms >= DB_SLOW_QUERY_MS:
                record_db_slow_query(getattr(context, _OPERATION_ATTR, None))
        _emit_pool_metrics(conn.engine.pool)

    # insert=True: must run before the instrumentor's hook ends the span.
    @event.listens_for(engine, "after_cursor_execute", insert=True)
    def _record_db_rows(
        conn: Connection,
        cursor: DBAPICursor,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        del conn, statement, parameters, executemany
        span = getattr(context, "_otel_span", None)
        rowcount = getattr(cursor, "rowcount", None)
        if span is None or rowcount is None or rowcount < 0:
            return
        if span.is_recording():
            span.set_attribute(DB_ROWS_ATTRIBUTE, int(rowcount))

    @event.listens_for(engine, "handle_error")
    def _handle_error(exception_context: ExceptionContext) -> None:
        context = exception_context.execution_context
        operation = (
            getattr(context, _OPERATION_ATTR, None) if context is not None else None
        )
        sqlstate = getattr(exception_context.original_exception, "sqlstate", None)
        _tag_db_error(sqlstate)
        record_db_query_failed(operation=operation, sqlstate=sqlstate)

    @event.listens_for(engine.pool, "checkout")
    def _on_checkout(
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
        connection_proxy: PoolProxiedConnection,
    ) -> None:
        del dbapi_connection, connection_record, connection_proxy
        _emit_pool_metrics(engine.pool)

    @event.listens_for(engine.pool, "checkin")
    def _on_checkin(
        dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
    ) -> None:
        del dbapi_connection, connection_record
        _emit_pool_metrics(engine.pool)

    @event.listens_for(engine.pool, "connect")
    def _on_connect(
        dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
    ) -> None:
        del dbapi_connection, connection_record
        record_db_connection_event("opened")

    @event.listens_for(engine.pool, "close")
    def _on_close(
        dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry
    ) -> None:
        del dbapi_connection, connection_record
        record_db_connection_event("closed")

    @event.listens_for(engine.pool, "invalidate")
    def _on_invalidate(
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
        exception: BaseException | None,
    ) -> None:
        del dbapi_connection, connection_record, exception
        record_db_connection_event("invalidated")

    @event.listens_for(engine, "commit")
    def _on_commit(conn: Connection) -> None:
        del conn
        record_db_transaction("commit")

    @event.listens_for(engine, "rollback")
    def _on_rollback(conn: Connection) -> None:
        del conn
        record_db_transaction("rollback")

    setattr(engine, _INSTRUMENTED_ATTR, True)
    logger.debug("[instrument_db_engine] SQLAlchemy DB telemetry enabled")
