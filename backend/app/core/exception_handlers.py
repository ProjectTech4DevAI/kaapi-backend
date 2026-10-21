import logging
import re
from collections import defaultdict

import sentry_sdk
from asgi_correlation_id import correlation_id
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DataError, IntegrityError, OperationalError, SQLAlchemyError
from starlette.status import (
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from app.utils import APIResponse

logger = logging.getLogger(__name__)

GENERIC_ERROR_DETAIL = "An unexpected error occurred."
DB_ERROR_DETAIL = "Database error: request could not be completed."

UNIQUE_VIOLATION_SQLSTATE = "23505"
FOREIGN_KEY_VIOLATION_SQLSTATE = "23503"

# Other IntegrityError codes (e.g. not-null 23502) mean our write was wrong: 500.
_INTEGRITY_ERROR_HINTS: dict[str, tuple[int, str]] = {
    UNIQUE_VIOLATION_SQLSTATE: (
        HTTP_409_CONFLICT,
        "Database error: request conflicts with existing data.",
    ),
    FOREIGN_KEY_VIOLATION_SQLSTATE: (
        HTTP_422_UNPROCESSABLE_CONTENT,
        "Database error: request references a record that does not exist.",
    ),
}

# Unique violations on these constraints get a specific message; values stay out of it.
_UNIQUE_CONSTRAINT_DETAILS: dict[str, str] = {
    "uq_config_project_id_name_active": "A config with this name already exists in this project.",
    "uq_config_version_config_id_version": "A version with this number already exists for this config.",
    "uq_credential_org_project_provider": "Credentials for this provider already exist for this organization and project.",
    "ix_organization_name": "An organization with this name already exists.",
    "uq_project_name_org_id": "A project with this name already exists in this organization.",
    "ix_user_email": "A user with this email already exists.",
}

_BRANCH_PATTERN = re.compile(r"^[A-Z]|[\[\]()]")


def _is_branch_identifier(part: str) -> bool:
    return bool(part and isinstance(part, str) and _BRANCH_PATTERN.search(part))


def _sanitize_validation_errors(errors: list[dict]) -> list[dict]:
    """Sanitize pydantic validation errors.

    Filters union branch noise (keeps only the relevant branch),
    strips internal fields, returning only loc, msg, and type.
    """
    try:
        branch_errors: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        non_union_errors: list[dict] = []

        for err in errors:
            loc = err.get("loc", ())
            branch_name = None
            parent_field = None
            for i, part in enumerate(loc):
                if _is_branch_identifier(part):
                    branch_name = part
                    parent_field = loc[:i] if i > 0 else ("root",)
                    break

            if branch_name and parent_field:
                branch_errors[str(parent_field)][branch_name].append(err)
            else:
                non_union_errors.append(err)

        filtered = list(non_union_errors)
        for _parent, branches in branch_errors.items():
            if len(branches) <= 1:
                for errs in branches.values():
                    filtered.extend(errs)
            else:
                # NOTE: Keep all branches tied for fewest literal errors
                best_count = min(
                    sum(1 for e in errs if e.get("type") == "literal_error")
                    for errs in branches.values()
                )
                for errs in branches.values():
                    if (
                        sum(1 for e in errs if e.get("type") == "literal_error")
                        <= best_count
                    ):
                        filtered.extend(errs)

        for err in filtered:
            loc = err.get("loc", ())
            err["loc"] = tuple(p for p in loc if not _is_branch_identifier(p))

        seen_errors: set[tuple] = set()
        unique_errors: list[dict] = []
        for error in filtered:
            error_key = (tuple(error.get("loc", ())), error.get("msg", ""))
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                unique_errors.append(error)

        sanitized = [
            {k: v for k, v in err.items() if k in ("loc", "msg", "type")}
            for err in (unique_errors or errors)
        ]
        return sanitized
    except Exception:
        return errors


def _db_error_constraint(exc: SQLAlchemyError) -> tuple[str | None, str | None]:
    """Postgres SQLSTATE and violated constraint name from the driver error, when present."""
    orig = getattr(exc, "orig", None)
    diag = getattr(orig, "diag", None)
    return getattr(orig, "sqlstate", None), getattr(diag, "constraint_name", None)


def _db_error_response(exc: SQLAlchemyError) -> tuple[int, str]:
    """Map a DB exception to a status and a client-safe hint; SQL and parameters stay server-side."""
    if isinstance(exc, IntegrityError):
        sqlstate, constraint = _db_error_constraint(exc)
        if (
            sqlstate == UNIQUE_VIOLATION_SQLSTATE
            and constraint in _UNIQUE_CONSTRAINT_DETAILS
        ):
            return HTTP_409_CONFLICT, _UNIQUE_CONSTRAINT_DETAILS[constraint]
        return _INTEGRITY_ERROR_HINTS.get(
            sqlstate, (HTTP_500_INTERNAL_SERVER_ERROR, DB_ERROR_DETAIL)
        )
    if isinstance(exc, DataError):
        return (
            HTTP_422_UNPROCESSABLE_CONTENT,
            "Database error: a value is invalid or too long.",
        )
    if isinstance(exc, OperationalError):
        return (
            HTTP_503_SERVICE_UNAVAILABLE,
            "Database error: service temporarily unavailable, please retry.",
        )
    return HTTP_500_INTERNAL_SERVER_ERROR, DB_ERROR_DETAIL


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = _sanitize_validation_errors(exc.errors())
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            content=APIResponse.failure_response(errors).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, list):
            detail = _sanitize_validation_errors(detail)

        # Tag at the boundary: the status is only known here, and call-site logs
        # fire before it. 5xx is a server fault (error event); 4xx is the caller's.
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("http.status_code", str(exc.status_code))
            sentry_sdk.set_tag(
                "error.class", "server" if exc.status_code >= 500 else "client"
            )
        if exc.status_code >= 500:
            logger.error(
                f"[http_exception] {request.method} {request.url.path} "
                f"-> {exc.status_code}",
                exc_info=exc,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.failure_response(detail).model_dump(),
        )

    @app.exception_handler(SQLAlchemyError)
    async def db_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        status_code, hint = _db_error_response(exc)
        sqlstate, constraint = _db_error_constraint(exc)
        if sentry_sdk.get_client().is_active():
            sentry_sdk.set_tag("http.status_code", str(status_code))
            sentry_sdk.set_tag(
                "error.class", "server" if status_code >= 500 else "client"
            )
            if constraint:
                sentry_sdk.set_tag("db.constraint", constraint)
        log = logger.error if status_code >= 500 else logger.warning
        log(
            f"[db_error_handler] Unhandled DB error | path: {request.url.path}, "
            f"type: {type(exc).__name__}, status: {status_code}, sqlstate: {sqlstate}, "
            f"constraint: {constraint}, trace_id: {correlation_id.get() or 'N/A'}",
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status_code,
            content=APIResponse.failure_response(hint).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Fixed detail: str(exc) here leaked provider bodies and DB text to callers.
        # logger.error alone reports it; adding capture_exception makes a second event.
        logger.error(
            f"[generic_error_handler] Unhandled exception | "
            f"path: {request.url.path}, trace_id: {correlation_id.get() or 'N/A'}",
            exc_info=exc,
        )
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.failure_response(GENERIC_ERROR_DETAIL).model_dump(),
        )
