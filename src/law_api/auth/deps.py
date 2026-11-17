"""FastAPI authentication and RBAC dependencies."""

from __future__ import annotations

from fastapi import Request

from ..exceptions import AuthenticationError, PermissionDeniedError
from ..lifecycle import ensure_services
from .models import TenantContext
from .security import decode_jwt


async def get_tenant_context(request: Request) -> TenantContext:
    """Resolve TenantContext from Bearer JWT or X-API-Key.

    When `settings.auth_required` is False and no credentials are supplied,
    falls back to the default workspace tenant so local demos and backward-
    compatible callers continue working seamlessly. Whenever credentials ARE
    supplied (or `auth_required=True`), they are strictly validated.
    """
    await ensure_services(request.app)
    store = request.app.state.tenant_store
    settings = request.app.state.settings

    auth_header = request.headers.get("Authorization", "").strip()
    api_key_header = request.headers.get("X-API-Key", "").strip()

    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        payload = decode_jwt(token, settings.jwt_secret_key)
        if not payload or not payload.get("sub"):
            raise AuthenticationError("Invalid or expired Bearer token")
        ctx = store.get_tenant_context_for_user(str(payload["sub"]), auth_type="jwt")
        if ctx is None:
            raise AuthenticationError("User associated with token no longer exists")
        request.state.tenant_context = ctx
        return ctx

    if api_key_header:
        ctx = store.authenticate_api_key(api_key_header)
        if ctx is None:
            raise AuthenticationError("Invalid or revoked API key")
        request.state.tenant_context = ctx
        return ctx

    if settings.auth_required:
        raise AuthenticationError("Missing Bearer token or X-API-Key header")

    ctx = store.get_default_context()
    request.state.tenant_context = ctx
    return ctx


def require_role(*allowed_roles: str):
    """Dependency factory enforcing RBAC role checks."""

    async def _checker(request: Request) -> TenantContext:
        ctx = await get_tenant_context(request)
        if not ctx.has_role(*allowed_roles):
            raise PermissionDeniedError(
                f"Role '{ctx.role}' lacks required permission ({', '.join(allowed_roles)})"
            )
        return ctx

    return _checker

