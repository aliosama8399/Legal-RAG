"""Identity and tenancy models."""

from dataclasses import dataclass

ROLE_HIERARCHY: dict[str, int] = {
    "viewer": 1,
    "attorney": 2,
    "admin": 3,
    "owner": 4,
}

PLAN_QUOTAS: dict[str, dict[str, int]] = {
    "free": {"monthly_tokens": 250_000, "max_documents": 10, "rpm": 60},
    "pro": {"monthly_tokens": 5_000_000, "max_documents": 250, "rpm": 300},
    "enterprise": {"monthly_tokens": 100_000_000, "max_documents": 10_000, "rpm": 1200},
}


@dataclass(frozen=True)
class TenantContext:
    """Resolved security & tenancy context attached to every request."""

    tenant_id: str
    user_id: str
    email: str
    role: str
    plan: str = "enterprise"
    org_name: str = "Default Organization"
    auth_type: str = "anonymous"

    def has_role(self, *allowed: str) -> bool:
        if self.role in allowed:
            return True
        min_required = min((ROLE_HIERARCHY.get(r, 99) for r in allowed), default=99)
        return ROLE_HIERARCHY.get(self.role, 0) >= min_required

