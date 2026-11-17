"""Enterprise domain exceptions formatted as RFC 9457 Problem Details."""

from __future__ import annotations


class AppError(Exception):
    """Base application error mapped cleanly to HTTP responses."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        title: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.title = title
        self.detail = detail
        self.headers = headers or {}


class AuthenticationError(AppError):
    def __init__(self, detail: str = "Authentication credentials were missing or invalid") -> None:
        super().__init__(
            401,
            "AUTHENTICATION_REQUIRED",
            "Authentication Failed",
            detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class PermissionDeniedError(AppError):
    def __init__(self, detail: str = "You do not have permission to perform this action") -> None:
        super().__init__(403, "PERMISSION_DENIED", "Forbidden", detail)


class ResourceNotFoundError(AppError):
    def __init__(self, detail: str = "Requested resource was not found") -> None:
        super().__init__(404, "RESOURCE_NOT_FOUND", "Not Found", detail)


class ConflictError(AppError):
    def __init__(self, detail: str = "Resource conflict") -> None:
        super().__init__(409, "CONFLICT", "Conflict", detail)


class ValidationAppError(AppError):
    def __init__(self, detail: str = "Validation failed") -> None:
        super().__init__(422, "VALIDATION_ERROR", "Unprocessable Entity", detail)


class RateLimitExceededError(AppError):
    def __init__(self, retry_after: int = 60) -> None:
        super().__init__(
            429,
            "RATE_LIMIT_EXCEEDED",
            "Too Many Requests",
            f"Rate limit exceeded. Retry in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


class QuotaExceededError(AppError):
    def __init__(self, detail: str = "Organization quota exceeded for current billing cycle") -> None:
        super().__init__(402, "QUOTA_EXCEEDED", "Quota Exceeded", detail)


class UpstreamServiceError(AppError):
    def __init__(self, detail: str = "Upstream inference or storage service unavailable") -> None:
        super().__init__(503, "UPSTREAM_UNAVAILABLE", "Service Unavailable", detail)

