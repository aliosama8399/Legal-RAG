"""Enterprise request-context, sliding-window rate-limiting, and RFC 9457 error middleware."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .exceptions import AppError

logger = logging.getLogger(__name__)

_EXEMPT_PATHS = frozenset(
    {
        "/api/v1/health",
        "/api/v1/health/ready",
        "/api/v1/metrics",
        "/healthz",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


class SlidingWindowRateLimiter:
    """In-memory sliding-window (60s) rate limiter per client/credential key."""

    def __init__(self, requests_per_minute: int = 120) -> None:
        self.rpm = max(1, requests_per_minute)
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, int]:
        """Return (allowed, remaining, reset_epoch_seconds)."""
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.rpm:
                oldest = bucket[0] if bucket else now
                reset_at = int(oldest + 60.0) + 1
                return False, 0, reset_at
            bucket.append(now)
            remaining = max(0, self.rpm - len(bucket))
            reset_at = int((bucket[0] if bucket else now) + 60.0) + 1
            return True, remaining, reset_at


def _rate_limit_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "").strip()
    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"key:{digest}"
    if auth.lower().startswith("bearer "):
        digest = hashlib.sha256(auth[7:].strip().encode("utf-8")).hexdigest()[:16]
        return f"jwt:{digest}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def _problem_payload(
    *,
    status: int,
    title: str,
    detail: str,
    error_code: str,
    request_id: str,
) -> dict:
    return {
        "type": f"https://docs.legalrag.io/errors/{error_code.lower()}",
        "title": title,
        "status": status,
        "error_code": error_code,
        "detail": detail,
        "request_id": request_id,
    }


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attaches X-Request-ID, enforces RPM rate limits, and sets security headers."""

    def __init__(self, app, rate_limit_rpm: int = 120) -> None:
        super().__init__(app)
        self.limiter = SlidingWindowRateLimiter(rate_limit_rpm)

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        started = time.perf_counter()

        path = request.url.path
        rate_headers: dict[str, str] = {}
        if path not in _EXEMPT_PATHS:
            allowed, remaining, reset_at = self.limiter.check(_rate_limit_key(request))
            rate_headers = {
                "X-RateLimit-Limit": str(self.limiter.rpm),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_at),
            }
            if not allowed:
                retry_after = max(1, reset_at - int(time.time()))
                body = _problem_payload(
                    status=429,
                    title="Too Many Requests",
                    detail=f"Rate limit of {self.limiter.rpm} requests/min exceeded.",
                    error_code="RATE_LIMIT_EXCEEDED",
                    request_id=request_id,
                )
                return JSONResponse(
                    status_code=429,
                    content=body,
                    headers={
                        "X-Request-ID": request_id,
                        "Retry-After": str(retry_after),
                        **rate_headers,
                    },
                )

        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        for k, v in rate_headers.items():
            response.headers[k] = v

        tenant_ctx = getattr(request.state, "tenant_context", None)
        if tenant_ctx is not None:
            response.headers["X-Tenant-ID"] = tenant_ctx.tenant_id

        logger.info(
            "http_request method=%s path=%s status=%s duration_ms=%s request_id=%s tenant_id=%s",
            request.method,
            path,
            response.status_code,
            duration_ms,
            request_id,
            tenant_ctx.tenant_id if tenant_ctx else "-",
        )
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Install RFC 9457 Problem Details handlers while preserving top-level `detail`."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        return JSONResponse(
            status_code=exc.status_code,
            content=_problem_payload(
                status=exc.status_code,
                title=exc.title,
                detail=exc.detail,
                error_code=exc.error_code,
                request_id=request_id,
            ),
            headers={"X-Request-ID": request_id, **exc.headers},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_problem_payload(
                status=exc.status_code,
                title=f"HTTP {exc.status_code}",
                detail=detail,
                error_code=f"HTTP_{exc.status_code}",
                request_id=request_id,
            ),
            headers={"X-Request-ID": request_id, **(exc.headers or {})},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        logger.exception("Unhandled error request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_problem_payload(
                status=500,
                title="Internal Server Error",
                detail="An unexpected internal error occurred. Reference X-Request-ID for support.",
                error_code="INTERNAL_ERROR",
                request_id=request_id,
            ),
            headers={"X-Request-ID": request_id},
        )

