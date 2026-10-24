"""Telemetry attribute keys and metric names: one name per concept across spans, tags, metrics and logs.

Tenant ids are always sent as strings so a single Sentry filter matches spans, errors and logs.
"""

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (  # noqa: F401  (re-exported)
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
)

from opentelemetry.semconv._incubating.attributes.db_attributes import (  # noqa: F401  (re-exported)
    DB_SYSTEM,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (  # noqa: F401  (re-exported)
    HTTP_REQUEST_BODY_SIZE,
)
from opentelemetry.semconv._incubating.attributes.user_attributes import (  # noqa: F401  (re-exported)
    USER_ID,
)

# Sentry AI Insights keys with no OTel semconv equivalent.
GEN_AI_REQUEST_AVAILABLE_TOOLS = "gen_ai.request.available_tools"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

TENANT_ORG_ID = "tenant.org_id"
TENANT_PROJECT_ID = "tenant.project_id"

CORRELATION_ID = "correlation_id"
JOB_ID = "job.id"
CELERY_TASK_ID = "celery.task_id"
CELERY_TASK_NAME = "celery.task_name"

LLM_CALL_ID = "llm.call_id"
LLM_CHAIN_ID = "llm.chain_id"
LLM_COMPLETION_TYPE = "llm.completion_type"

HTTP_METHOD = "http.method"
HTTP_ROUTE = "http.route"
HTTP_STATUS_CODE = "http.status_code"
HTTP_ROUTE_CATEGORY = "http.route_category"
ERROR_CLASS = "error.class"
ALERT_TYPE = "alert.type"
SENTRY_OP = "sentry.op"

LLM_CONFIG_ID = "llm.config.id"
LLM_CONFIG_VERSION = "llm.config.version"
LLM_CONFIG_IS_STORED = "llm.config.is_stored"
LLM_CALLBACK_ENABLED = "llm.callback_enabled"
LLM_PROXY_URL = "llm.proxy.url"

CALLBACK_URL = "callback.url"
CALLBACK_STATUS = "callback.status"
CALLBACK_REQUEST_BODY_SIZE = "callback.request.body.size"
CALLBACK_RESPONSE_STATUS_CODE = "callback.response.status_code"
CALLBACK_DURATION_MS = "callback.duration_ms"

COLLECTION_ID = "collection.id"
COLLECTION_PROVIDER = "collection.provider"

JOB_TABLE = "job.table"
JOB_STATUS = "job.status"
JOB_TYPE = "job.type"
JOB_ACTION_TYPE = "job.action_type"

# OTel's stable name for SQLSTATE is db.response.status_code; kept as-is so existing dashboards keep working.
DB_SQLSTATE = "db.sqlstate"
DB_ERROR_NAME = "db.error.name"
DB_CONSTRAINT = "db.constraint"
DB_OPERATION = "db.operation"
DB_ROWS_AFFECTED = "db.rows_affected"

METRIC_HTTP_REQUEST_COUNT = "http.server.request.count"
METRIC_HTTP_REQUEST_DURATION = "http.server.request.duration"
METRIC_HTTP_REQUEST_BODY_SIZE = "http.server.request.body.size"
METRIC_HTTP_REQUEST_ERROR = "http.server.request.error"
METRIC_HTTP_REQUEST_UNMATCHED = "http.server.request.unmatched"
METRIC_LLM_CALL_COUNT = "llm.call.count"
METRIC_LLM_CALL_ERROR = "llm.call.error"
METRIC_LLM_CALL_DURATION = "llm.call.duration"
METRIC_LLM_TOKENS_INPUT = "llm.tokens.input"
METRIC_LLM_TOKENS_OUTPUT = "llm.tokens.output"
METRIC_LLM_TOKENS_TOTAL = "llm.tokens.total"
METRIC_DB_QUERY_FAILED = "db.query.failed"
METRIC_DB_QUERY_SLOW = "db.query.slow"
METRIC_DB_CONNECTION_OPENED = "db.connection.opened"
METRIC_DB_CONNECTION_CLOSED = "db.connection.closed"
METRIC_DB_CONNECTION_INVALIDATED = "db.connection.invalidated"
METRIC_DB_TRANSACTION_COMMIT = "db.transaction.commit"
METRIC_DB_TRANSACTION_ROLLBACK = "db.transaction.rollback"
METRIC_DB_POOL_ACTIVE = "db.pool.active"
METRIC_DB_POOL_IDLE = "db.pool.idle"
METRIC_DB_POOL_TOTAL = "db.pool.total"
METRIC_DB_POOL_OVERFLOW = "db.pool.overflow"
# Unit in the name predates the naming rule; renaming would break existing dashboards.
METRIC_JOB_PENDING_STALE_COUNT = "jobs.pending.stale.count"
METRIC_JOB_PENDING_STALE_BY_DIMENSION_COUNT = "jobs.pending.stale.by_dimension.count"
METRIC_JOB_PENDING_OLDEST_AGE = "jobs.pending.oldest_age_seconds"
METRIC_JOB_PENDING_OLDEST_AGE_BY_DIMENSION = (
    "jobs.pending.oldest_age_seconds.by_dimension"
)
