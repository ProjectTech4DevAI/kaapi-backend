"""Postgres checkpointer for the eval-iterate-improve LangGraph loop.

`langgraph-checkpoint-postgres` connects via psycopg (v3) directly rather than
through the app's SQLAlchemy engine, so it owns its own small pool and its own
tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) — not Alembic
managed.
"""

import logging
from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 5


def _psycopg_conn_string() -> str:
    """Derive a plain psycopg conninfo string from the app's SQLAlchemy DSN.

    The app's DSN already targets the psycopg driver (`postgresql+psycopg://`),
    so stripping the SQLAlchemy dialect qualifier is the only adaptation needed.
    """
    return str(settings.SQLALCHEMY_DATABASE_URI).replace(
        "postgresql+psycopg://", "postgresql://", 1
    )


@lru_cache(maxsize=1)
def get_evaluation_iteration_checkpointer() -> PostgresSaver:
    """Module-level singleton checkpointer, backed by its own connection pool.

    `.setup()` is `CREATE TABLE IF NOT EXISTS`-style, so it is safe to run on
    first access rather than behind a separate startup hook.
    """
    pool = ConnectionPool(
        conninfo=_psycopg_conn_string(),
        min_size=_POOL_MIN_SIZE,
        max_size=_POOL_MAX_SIZE,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    logger.info("[get_evaluation_iteration_checkpointer] Checkpointer ready")
    return checkpointer
