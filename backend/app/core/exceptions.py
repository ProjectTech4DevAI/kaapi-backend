"""Domain exceptions carrying the status code they surface as.

Raise from crud/service instead of a bare `ValueError`, which falls through to the
generic 500 handler. Mapped to responses by `app.core.exception_handlers`.
"""

from starlette import status


class KaapiError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(KaapiError):
    status_code = status.HTTP_404_NOT_FOUND


class ConflictError(KaapiError):
    status_code = status.HTTP_409_CONFLICT


class InvalidValueError(KaapiError):
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidPayloadError(KaapiError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class UpstreamError(KaapiError):
    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, detail: str, provider: str) -> None:
        super().__init__(detail)
        self.provider = provider


class ServiceUnavailableError(KaapiError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
