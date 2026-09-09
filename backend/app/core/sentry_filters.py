import re
from typing import Any


_REDACTED = "[REDACTED]"

# sentry_sdk's CeleryIntegration attaches the task's raw args/kwargs to every
# event captured during that task as extra["celery-job"]. For these tasks,
# kwargs["request_data"] carries the caller's raw query text and (for chain
# jobs) prior block responses -- none of that should reach Sentry.
_LLM_JOB_TASK_NAMES = {
    "app.celery.tasks.job_execution.run_llm_job",
    "app.celery.tasks.job_execution.run_llm_chain_job",
    "app.celery.tasks.job_execution.run_response_job",
}
_SENSITIVE_REQUEST_DATA_KEYS = ("query", "request_metadata", "response", "output")


_SQL_OR_CONNECT = re.compile(r"^(select|insert|update|delete|connect)\b", re.IGNORECASE)
_HTTP_SEND_RECEIVE = re.compile(r"http (send|receive)$", re.IGNORECASE)
_DB_QUERY_SPAN = re.compile(r"^db\.query$", re.IGNORECASE)
_BARE_HTTP_METHOD = re.compile(
    r"^(GET|HEAD|OPTIONS|POST|PUT|PATCH|DELETE|TRACE|CONNECT)$", re.IGNORECASE
)
_NOISE_PATH = re.compile(
    r"(^/health/?$|^/robots\.txt$|^/favicon\.ico$|^/wp-admin|^/wp-login|^/xmlrpc\.php$)",
    re.IGNORECASE,
)


def _extract_path(event: dict[str, Any]) -> str:
    request = event.get("request")
    if not isinstance(request, dict):
        return ""

    url = request.get("url")
    if isinstance(url, str) and url:
        if "://" in url:
            after_scheme = url.split("://", 1)[1]
            if "/" in after_scheme:
                return "/" + after_scheme.split("/", 1)[1].split("?", 1)[0]
            return "/"
        return url.split("?", 1)[0]
    return ""


def _should_drop_transaction(event: dict[str, Any]) -> bool:
    transaction = str(event.get("transaction") or "").strip()
    path = _extract_path(event)

    # Sentry shows probe traffic as bare "GET"/"HEAD" transactions.
    if _BARE_HTTP_METHOD.match(transaction):
        return True

    # Drop known non-app noise paths (probes / scanners).
    if path and _NOISE_PATH.search(path):
        return True

    return False


def before_send_transaction_filter(
    event: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Drop low-signal spans before they ship to Sentry.

    Filters out:
    - ASGI lifecycle spans ending with `http send` / `http receive`
    - DB spans carrying `db.system`
    - SQL / `connect` spans matched by description prefix
    - Custom DB query spans (`db.query`)
    """
    if _should_drop_transaction(event):
        return None

    spans = event.get("spans")
    if not isinstance(spans, list):
        return event

    filtered: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue

        data = span.get("data") if isinstance(span.get("data"), dict) else {}
        desc = str(span.get("description") or span.get("name") or "").strip()
        op = str(span.get("op") or "").strip()

        if _HTTP_SEND_RECEIVE.search(desc):
            continue
        if _DB_QUERY_SPAN.search(desc) or _DB_QUERY_SPAN.search(op):
            continue
        if data.get("db.system") is not None:
            continue
        if _SQL_OR_CONNECT.match(desc):
            continue

        filtered.append(span)

    event["spans"] = filtered
    return event


def _redact_llm_job_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replaces end-user text fields in an LLM job's request_data with a
    placeholder, leaving non-text fields (config, ids, callback_url) intact.
    """
    request_data = kwargs.get("request_data")
    if not isinstance(request_data, dict):
        return kwargs

    redacted_request_data = dict(request_data)
    for key in _SENSITIVE_REQUEST_DATA_KEYS:
        if key in redacted_request_data:
            redacted_request_data[key] = _REDACTED

    redacted_kwargs = dict(kwargs)
    redacted_kwargs["request_data"] = redacted_request_data
    return redacted_kwargs


def before_send_filter(
    event: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Strips end-user query/response text from LLM job celery-job context
    before an event reaches Sentry.
    """
    extra = event.get("extra")
    if not isinstance(extra, dict):
        return event

    celery_job = extra.get("celery-job")
    if not isinstance(celery_job, dict):
        return event

    if celery_job.get("task_name") not in _LLM_JOB_TASK_NAMES:
        return event

    kwargs = celery_job.get("kwargs")
    if isinstance(kwargs, dict):
        celery_job["kwargs"] = _redact_llm_job_kwargs(kwargs)

    return event
