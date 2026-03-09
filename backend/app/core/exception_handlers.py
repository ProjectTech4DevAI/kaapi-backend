import re
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from app.utils import APIResponse

_BRANCH_PATTERN = re.compile(r"^[A-Z]|[\[\]()]")


def _is_branch_identifier(part: str) -> bool:
    return bool(part and isinstance(part, str) and _BRANCH_PATTERN.search(part))


def _filter_union_branch_errors(errors: list[dict]) -> list[dict]:
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
                best_branch = min(
                    branches.items(),
                    key=lambda item: (
                        sum(1 for e in item[1] if e.get("type") == "literal_error"),
                    ),
                )
                filtered.extend(best_branch[1])

        for err in filtered:
            loc = err.get("loc", ())
            err["loc"] = tuple(p for p in loc if not _is_branch_identifier(p))

        return filtered or errors
    except Exception:
        return errors


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = _filter_union_branch_errors(exc.errors())
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            content=APIResponse.failure_response(errors).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.failure_response(exc.detail).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.failure_response(
                str(exc) or "An unexpected error occurred."
            ).model_dump(),
        )
