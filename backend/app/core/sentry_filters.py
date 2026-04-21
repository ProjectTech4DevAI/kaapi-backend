import re
from typing import Any


_SQL_OR_CONNECT = re.compile(r"^(select|insert|update|delete|connect)\b", re.IGNORECASE)
_HTTP_SEND_RECEIVE = re.compile(r"http (send|receive)$", re.IGNORECASE)
_DB_QUERY_SPAN = re.compile(r"^db\.query$", re.IGNORECASE)


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
