"""Telemetry keys shared across modules, so every span, Sentry tag and metric uses one name per concept.

Only keys used in more than one place, or that dashboards and alerts filter on, live here.
A key used in a single file stays as a string at that call site.
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

# --- Tenant: always sent as strings so one filter matches spans, errors and logs ---
TENANT_ORG_ID = "tenant.org_id"
TENANT_PROJECT_ID = "tenant.project_id"

# --- Request and job identity ---
CORRELATION_ID = "correlation_id"
JOB_ID = "job.id"

# --- GenAI: OTel semconv keys above, plus Sentry AI Insights keys semconv doesn't define ---
GEN_AI_REQUEST_AVAILABLE_TOOLS = "gen_ai.request.available_tools"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

# --- HTTP ---
HTTP_METHOD = "http.method"
HTTP_ROUTE = "http.route"
HTTP_STATUS_CODE = "http.status_code"

# --- Database: DB_SYSTEM comes from OTel semconv (imported above) ---
DB_SQLSTATE = "db.sqlstate"
DB_ERROR_NAME = "db.error.name"
DB_CONSTRAINT = "db.constraint"
DB_OPERATION = "db.operation"
DB_ROWS_AFFECTED = "db.rows_affected"

# --- Logs: fields added to every log record ---
# Sentry skips "_"-prefixed record attributes, so only LOG_MODULE_FIELD reaches Sentry Logs.
LOG_MODULE_FIELD = "log_module"
LOG_LOCATION_FIELD = "_log_location"

# --- Errors and alerts ---
ERROR_CLASS = "error.class"
ALERT_TYPE = "alert.type"

# --- Outbound callbacks (webhooks to the client's callback_url) ---
CALLBACK_URL = "callback.url"
CALLBACK_ENABLED = "callback.enabled"
CALLBACK_STATUS = "callback.status"
CALLBACK_REQUEST_BODY_SIZE = "callback.request.body.size"
CALLBACK_RESPONSE_STATUS_CODE = "callback.response.status_code"
CALLBACK_DURATION_MS = "callback.duration_ms"

# --- Cron: counts on the spans a cron run opens when it finds work ---
CRON_RUNS_PENDING = "cron.runs.pending"
CRON_RUNS_PROCESSED = "cron.runs.processed"
CRON_RUNS_FAILED = "cron.runs.failed"
CRON_RUNS_STILL_PROCESSING = "cron.runs.still_processing"
CRON_RUNS_REAPED = "cron.runs.reaped"
CRON_AGGREGATES_DISPATCHED = "cron.aggregates_dispatched"
CRON_CHUNKS_REENQUEUED = "cron.chunks_reenqueued"
CRON_RESUMES_DISPATCHED = "cron.resumes_dispatched"
