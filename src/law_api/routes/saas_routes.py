"""Enterprise SaaS routes: auth, organizations, API keys, usage ledger, and audit logs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ..auth.deps import get_tenant_context, require_role
from ..auth.models import TenantContext
from ..auth.security import create_jwt
from ..lifecycle import ensure_services
from ..schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyListResponse,
    AuditLogResponse,
    AuthTokenResponse,
    LoginRequest,
    RegisterRequest,
    UsageSummaryResponse,
    UserProfileResponse,
)

saas_router = APIRouter(prefix="/api/v1")


def _profile_from_ctx(ctx: TenantContext) -> UserProfileResponse:
    return UserProfileResponse(
        tenant_id=ctx.tenant_id,
        org_name=ctx.org_name,
        user_id=ctx.user_id,
        email=ctx.email,
        role=ctx.role,
        plan=ctx.plan,
        auth_type=ctx.auth_type,
    )


@saas_router.post("/auth/register", response_model=AuthTokenResponse, tags=["auth"], status_code=201)
async def register(request: Request, body: RegisterRequest) -> AuthTokenResponse:
    """Create a new organization (tenant) + owner user and issue a Bearer JWT."""
    await ensure_services(request.app)
    store = request.app.state.tenant_store
    settings = request.app.state.settings

    created = store.register_organization(
        org_name=body.org_name,
        email=body.email,
        password=body.password,
        full_name=body.full_name,
        plan=body.plan,
    )
    token = create_jwt(
        {"sub": created["user_id"], "tid": created["tenant_id"], "role": created["role"]},
        settings.jwt_secret_key,
        settings.jwt_expire_minutes,
    )
    request_id = getattr(request.state, "request_id", "")
    ip_addr = request.client.host if request.client else ""
    store.record_audit_log(
        tenant_id=created["tenant_id"],
        actor_id=created["user_id"],
        action="organization.registered",
        resource_type="organization",
        resource_id=created["tenant_id"],
        metadata={"org_name": created["org_name"], "plan": created["plan"]},
        ip_address=ip_addr,
        request_id=request_id,
    )
    profile = UserProfileResponse(
        tenant_id=created["tenant_id"],
        org_name=created["org_name"],
        user_id=created["user_id"],
        email=created["email"],
        role=created["role"],
        plan=created["plan"],
        auth_type="jwt",
    )
    return AuthTokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        profile=profile,
    )


@saas_router.post("/auth/login", response_model=AuthTokenResponse, tags=["auth"])
async def login(request: Request, body: LoginRequest) -> AuthTokenResponse:
    """Authenticate an existing user and issue a Bearer JWT."""
    await ensure_services(request.app)
    store = request.app.state.tenant_store
    settings = request.app.state.settings

    ctx = store.authenticate_user(body.email, body.password)
    token = create_jwt(
        {"sub": ctx.user_id, "tid": ctx.tenant_id, "role": ctx.role},
        settings.jwt_secret_key,
        settings.jwt_expire_minutes,
    )
    request_id = getattr(request.state, "request_id", "")
    ip_addr = request.client.host if request.client else ""
    store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="auth.login",
        resource_type="user",
        resource_id=ctx.user_id,
        ip_address=ip_addr,
        request_id=request_id,
    )
    return AuthTokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        profile=_profile_from_ctx(ctx),
    )


@saas_router.get("/auth/me", response_model=UserProfileResponse, tags=["auth"])
async def me(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> UserProfileResponse:
    """Return the active tenant & user identity resolved from JWT or X-API-Key."""
    return _profile_from_ctx(ctx)


@saas_router.post(
    "/tenants/api-keys",
    response_model=ApiKeyCreatedResponse,
    tags=["tenants"],
    status_code=201,
)
async def create_api_key(
    request: Request,
    body: ApiKeyCreateRequest,
    ctx: Annotated[TenantContext, Depends(require_role("admin", "owner"))],
) -> ApiKeyCreatedResponse:
    """Mint a scoped API key (`lrag_live_...`). Plaintext is returned once."""
    store = request.app.state.tenant_store
    created = store.create_api_key(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        name=body.name,
        scopes=",".join(body.scopes),
    )
    store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="api_key.created",
        resource_type="api_key",
        resource_id=created["id"],
        metadata={"name": created["name"], "prefix": created["key_prefix"]},
        ip_address=request.client.host if request.client else "",
        request_id=getattr(request.state, "request_id", ""),
    )
    return ApiKeyCreatedResponse(**created)


@saas_router.get("/tenants/api-keys", response_model=ApiKeyListResponse, tags=["tenants"])
async def list_api_keys(
    request: Request,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> ApiKeyListResponse:
    """List masked API keys belonging to the active tenant."""
    store = request.app.state.tenant_store
    return ApiKeyListResponse(results=store.list_api_keys(ctx.tenant_id))


@saas_router.delete("/tenants/api-keys/{key_id}", status_code=204, tags=["tenants"])
async def revoke_api_key(
    request: Request,
    key_id: str,
    ctx: Annotated[TenantContext, Depends(require_role("admin", "owner"))],
) -> None:
    """Revoke an active API key immediately."""
    store = request.app.state.tenant_store
    store.revoke_api_key(ctx.tenant_id, key_id)
    store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="api_key.revoked",
        resource_type="api_key",
        resource_id=key_id,
        ip_address=request.client.host if request.client else "",
        request_id=getattr(request.state, "request_id", ""),
    )


@saas_router.get("/tenants/usage", response_model=UsageSummaryResponse, tags=["tenants"])
async def tenant_usage(
    request: Request,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> UsageSummaryResponse:
    """Return aggregated token metering, quota utilization, and cost attribution."""
    store = request.app.state.tenant_store
    return UsageSummaryResponse(**store.get_usage_summary(ctx.tenant_id))


@saas_router.get("/tenants/audit-logs", response_model=AuditLogResponse, tags=["tenants"])
async def tenant_audit_logs(
    request: Request,
    ctx: Annotated[TenantContext, Depends(require_role("admin", "owner"))],
    limit: int = 100,
) -> AuditLogResponse:
    """Return immutable SOC2-ready audit logs for the current organization."""
    store = request.app.state.tenant_store
    return AuditLogResponse(results=store.list_audit_logs(ctx.tenant_id, limit=limit))

