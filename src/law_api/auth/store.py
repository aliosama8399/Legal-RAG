"""Tenant, identity, API key, job, conversation, usage ledger, and audit store."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..exceptions import AuthenticationError, ConflictError, ResourceNotFoundError
from .models import PLAN_QUOTAS, TenantContext
from .security import generate_api_key, hash_api_key, hash_password, verify_password

_SCHEMA = """
CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free',
    monthly_token_quota INTEGER NOT NULL DEFAULT 250000,
    max_documents INTEGER NOT NULL DEFAULT 10,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'owner',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    scopes TEXT NOT NULL DEFAULT 'read,write',
    revoked_at TEXT,
    last_used_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents_meta (
    document_id INTEGER NOT NULL,
    tenant_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_hash TEXT NOT NULL DEFAULT '',
    article_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    embedding_model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, document_id)
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    document_id INTEGER,
    filename TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress_pct INTEGER NOT NULL DEFAULT 0,
    articles INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    document_id INTEGER,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    trace_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_ledger (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    request_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    ip_address TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    message_id TEXT NOT NULL DEFAULT '',
    score INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_tenant_idemp ON ingestion_jobs (tenant_id, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_conv_tenant ON conversations (tenant_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_usage_tenant ON usage_ledger (tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_logs (tenant_id, created_at);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    lowered = (model or "").lower()
    if "gpt-4o" in lowered and "mini" not in lowered:
        in_rate, out_rate = 2.50, 10.00
    elif "7b" in lowered or "14b" in lowered or "31b" in lowered:
        in_rate, out_rate = 0.25, 0.75
    else:
        in_rate, out_rate = 0.15, 0.60
    cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
    return round(cost, 8)


class TenantStore:
    """Thread-safe SQLite metadata & SaaS ledger store."""

    def __init__(self, db_path: Path, default_tenant_id: str = "org_default") -> None:
        self.db_path = Path(db_path)
        self.default_tenant_id = default_tenant_id
        self._lock = threading.RLock()
        self._init_db()

    @contextlib.contextmanager
    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            now = _now_iso()
            quota = PLAN_QUOTAS["enterprise"]
            conn.execute(
                """
                INSERT OR IGNORE INTO organizations
                (id, name, plan, monthly_token_quota, max_documents, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.default_tenant_id,
                    "Default Enterprise Workspace",
                    "enterprise",
                    quota["monthly_tokens"],
                    quota["max_documents"],
                    now,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO users
                (id, tenant_id, email, full_name, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "usr_default",
                    self.default_tenant_id,
                    "admin@legalrag.io",
                    "Workspace Admin",
                    hash_password("LegalRagAdmin!2026"),
                    "owner",
                    now,
                ),
            )

    def register_organization(
        self,
        *,
        org_name: str,
        email: str,
        password: str,
        full_name: str = "Admin",
        plan: str = "pro",
    ) -> dict:
        normalized_email = email.strip().lower()
        selected_plan = plan if plan in PLAN_QUOTAS else "pro"
        quotas = PLAN_QUOTAS[selected_plan]
        tenant_id = f"org_{uuid.uuid4().hex[:12]}"
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (normalized_email,)
            ).fetchone()
            if existing:
                raise ConflictError(f"A user with email '{normalized_email}' already exists")
            conn.execute(
                """
                INSERT INTO organizations
                (id, name, plan, monthly_token_quota, max_documents, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    org_name.strip(),
                    selected_plan,
                    quotas["monthly_tokens"],
                    quotas["max_documents"],
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO users
                (id, tenant_id, email, full_name, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    tenant_id,
                    normalized_email,
                    full_name.strip() or "Admin",
                    hash_password(password),
                    "owner",
                    now,
                ),
            )
        return {
            "tenant_id": tenant_id,
            "org_name": org_name.strip(),
            "plan": selected_plan,
            "user_id": user_id,
            "email": normalized_email,
            "full_name": full_name.strip() or "Admin",
            "role": "owner",
        }

    def authenticate_user(self, email: str, password: str) -> TenantContext:
        normalized_email = email.strip().lower()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id AS user_id, u.tenant_id, u.email, u.password_hash, u.role,
                       o.name AS org_name, o.plan
                FROM users u
                JOIN organizations o ON o.id = u.tenant_id
                WHERE u.email = ?
                """,
                (normalized_email,),
            ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise AuthenticationError("Invalid email or password")
        return TenantContext(
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            email=row["email"],
            role=row["role"],
            plan=row["plan"],
            org_name=row["org_name"],
            auth_type="jwt",
        )

    def get_tenant_context_for_user(self, user_id: str, auth_type: str = "jwt") -> TenantContext | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id AS user_id, u.tenant_id, u.email, u.role,
                       o.name AS org_name, o.plan
                FROM users u
                JOIN organizations o ON o.id = u.tenant_id
                WHERE u.id = ?
                """,
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return TenantContext(
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            email=row["email"],
            role=row["role"],
            plan=row["plan"],
            org_name=row["org_name"],
            auth_type=auth_type,
        )

    def get_default_context(self) -> TenantContext:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, plan FROM organizations WHERE id = ?",
                (self.default_tenant_id,),
            ).fetchone()
        return TenantContext(
            tenant_id=self.default_tenant_id,
            user_id="usr_default",
            email="admin@legalrag.io",
            role="owner",
            plan=row["plan"] if row else "enterprise",
            org_name=row["name"] if row else "Default Enterprise Workspace",
            auth_type="anonymous",
        )

    def create_api_key(
        self, *, tenant_id: str, user_id: str, name: str, scopes: str = "read,write"
    ) -> dict:
        plaintext, prefix, key_hash = generate_api_key()
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys
                (id, tenant_id, user_id, name, key_prefix, key_hash, scopes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key_id, tenant_id, user_id, name.strip(), prefix, key_hash, scopes, now),
            )
        return {
            "id": key_id,
            "name": name.strip(),
            "key_prefix": prefix,
            "api_key": plaintext,
            "scopes": scopes.split(","),
            "created_at": now,
        }

    def authenticate_api_key(self, plaintext: str) -> TenantContext | None:
        key_hash = hash_api_key(plaintext)
        now = _now_iso()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT k.id AS key_id, k.tenant_id, k.user_id, k.revoked_at,
                       u.email, u.role, o.name AS org_name, o.plan
                FROM api_keys k
                JOIN users u ON u.id = k.user_id
                JOIN organizations o ON o.id = k.tenant_id
                WHERE k.key_hash = ?
                """,
                (key_hash,),
            ).fetchone()
            if not row or row["revoked_at"] is not None:
                return None
            conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, row["key_id"]))
        return TenantContext(
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            email=row["email"],
            role=row["role"],
            plan=row["plan"],
            org_name=row["org_name"],
            auth_type="api_key",
        )

    def list_api_keys(self, tenant_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, key_prefix, scopes, revoked_at, last_used_at, created_at
                FROM api_keys
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "key_prefix": r["key_prefix"],
                "scopes": r["scopes"].split(","),
                "active": r["revoked_at"] is None,
                "revoked_at": r["revoked_at"],
                "last_used_at": r["last_used_at"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def revoke_api_key(self, tenant_id: str, key_id: str) -> None:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE api_keys SET revoked_at = ?
                WHERE id = ? AND tenant_id = ? AND revoked_at IS NULL
                """,
                (now, key_id, tenant_id),
            )
            if cur.rowcount == 0:
                raise ResourceNotFoundError(f"API key '{key_id}' not found")

    # ---- Documents & Ingestion Jobs ----

    def upsert_document_meta(
        self,
        *,
        tenant_id: str,
        document_id: int,
        filename: str,
        source_hash: str = "",
        article_count: int = 0,
        chunk_count: int = 0,
        status: str = "pending",
        embedding_model: str = "",
    ) -> None:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents_meta
                (document_id, tenant_id, filename, source_hash, article_count, chunk_count,
                 status, embedding_model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, document_id) DO UPDATE SET
                    filename = COALESCE(NULLIF(excluded.filename, ''), documents_meta.filename),
                    source_hash = COALESCE(NULLIF(excluded.source_hash, ''), documents_meta.source_hash),
                    article_count = CASE WHEN excluded.article_count > 0 THEN excluded.article_count ELSE documents_meta.article_count END,
                    chunk_count = CASE WHEN excluded.chunk_count > 0 THEN excluded.chunk_count ELSE documents_meta.chunk_count END,
                    status = excluded.status,
                    embedding_model = COALESCE(NULLIF(excluded.embedding_model, ''), documents_meta.embedding_model),
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    tenant_id,
                    filename,
                    source_hash,
                    article_count,
                    chunk_count,
                    status,
                    embedding_model,
                    now,
                    now,
                ),
            )

    def list_documents_meta(self, tenant_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, tenant_id, filename, source_hash, article_count, chunk_count,
                       status, embedding_model, created_at, updated_at
                FROM documents_meta
                WHERE tenant_id = ?
                ORDER BY document_id DESC
                """,
                (tenant_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document_meta(self, tenant_id: str, document_id: int) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents_meta WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_document_meta(self, tenant_id: str, document_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM documents_meta WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            return cur.rowcount > 0

    def create_ingestion_job(
        self,
        *,
        tenant_id: str,
        filename: str,
        idempotency_key: str | None = None,
    ) -> tuple[dict, bool]:
        """Create an ingestion job or return existing one when idempotency_key matches."""
        with self._lock, self._connect() as conn:
            if idempotency_key:
                existing = conn.execute(
                    """
                    SELECT * FROM ingestion_jobs
                    WHERE tenant_id = ? AND idempotency_key = ? AND status IN ('QUEUED', 'PROCESSING', 'READY')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (tenant_id, idempotency_key),
                ).fetchone()
                if existing:
                    return dict(existing), True

            job_id = f"job_{uuid.uuid4().hex[:16]}"
            now = _now_iso()
            conn.execute(
                """
                INSERT INTO ingestion_jobs
                (id, tenant_id, document_id, filename, idempotency_key, status, stage, progress_pct, created_at, updated_at)
                VALUES (?, ?, NULL, ?, ?, 'QUEUED', 'queued', 0, ?, ?)
                """,
                (job_id, tenant_id, filename, idempotency_key, now, now),
            )
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row), False

    def fail_stale_ingestion_jobs(
        self, active_ids: set[str] | None = None, reason: str = "Interrupted by container restart"
    ) -> int:
        """Mark leftover QUEUED/PROCESSING jobs with no running task as FAILED."""
        now = _now_iso()
        active = set(active_ids or ())
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM ingestion_jobs WHERE status IN ('QUEUED', 'PROCESSING')"
            ).fetchall()
            stale_ids = [row["id"] for row in rows if row["id"] not in active]
            if not stale_ids:
                return 0
            placeholders = ",".join("?" for _ in stale_ids)
            cur = conn.execute(
                f"""
                UPDATE ingestion_jobs
                SET status = 'FAILED', stage = 'failed', progress_pct = 0, error = ?, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                [reason, now, *stale_ids],
            )
            return cur.rowcount

    def reset_ingestion_job(self, job_id: str) -> dict:
        """Reset an orphaned job so it can be re-executed immediately."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'QUEUED', stage = 'queued', progress_pct = 0, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row)

    def update_ingestion_job(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        progress_pct: int,
        document_id: int | None = None,
        articles: int | None = None,
        chunks: int | None = None,
        error: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    stage = ?,
                    progress_pct = ?,
                    document_id = COALESCE(?, document_id),
                    articles = COALESCE(?, articles),
                    chunks = COALESCE(?, chunks),
                    error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (status, stage, progress_pct, document_id, articles, chunks, error, now, job_id),
            )

    def get_ingestion_job(self, tenant_id: str, job_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            ).fetchone()
        return dict(row) if row else None

    # ---- Multi-turn Conversations ----

    def create_conversation(
        self, *, tenant_id: str, user_id: str, title: str, document_id: int | None = None
    ) -> dict:
        conv_id = f"conv_{uuid.uuid4().hex[:14]}"
        now = _now_iso()
        clean_title = title.strip() or "New Legal Research Thread"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, tenant_id, user_id, document_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conv_id, tenant_id, user_id, document_id, clean_title, now, now),
            )
        return {
            "id": conv_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "document_id": document_id,
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
        }

    def list_conversations(self, tenant_id: str, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.conversation_id = c.id
                WHERE c.tenant_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (tenant_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, tenant_id: str, conversation_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            conv = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND tenant_id = ?",
                (conversation_id, tenant_id),
            ).fetchone()
            if not conv:
                return None
            msgs = conn.execute(
                """
                SELECT id, role, content, sources_json, trace_id, created_at
                FROM conversation_messages
                WHERE conversation_id = ? AND tenant_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id, tenant_id),
            ).fetchall()
        messages = [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "sources": json.loads(m["sources_json"] or "[]"),
                "trace_id": m["trace_id"],
                "created_at": m["created_at"],
            }
            for m in msgs
        ]
        return {**dict(conv), "messages": messages}

    def append_conversation_message(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        trace_id: str = "",
    ) -> dict:
        msg_id = f"msg_{uuid.uuid4().hex[:14]}"
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages
                (id, conversation_id, tenant_id, role, content, sources_json, trace_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    conversation_id,
                    tenant_id,
                    role,
                    content,
                    json.dumps(sources or [], ensure_ascii=False),
                    trace_id,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ? AND tenant_id = ?",
                (now, conversation_id, tenant_id),
            )
        return {
            "id": msg_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "trace_id": trace_id,
            "created_at": now,
        }

    def delete_conversation(self, tenant_id: str, conversation_id: str) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ? AND tenant_id = ?",
                (conversation_id, tenant_id),
            )
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND tenant_id = ?",
                (conversation_id, tenant_id),
            )
            return cur.rowcount > 0

    # ---- Usage Metering, Feedback & Audit Logs ----

    def record_usage(
        self,
        *,
        tenant_id: str,
        user_id: str,
        operation: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        request_id: str = "",
    ) -> None:
        entry_id = f"usg_{uuid.uuid4().hex[:14]}"
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_ledger
                (id, tenant_id, user_id, operation, model, prompt_tokens, completion_tokens,
                 estimated_cost_usd, request_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    tenant_id,
                    user_id,
                    operation,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    request_id,
                    now,
                ),
            )

    def get_usage_summary(self, tenant_id: str) -> dict:
        with self._lock, self._connect() as conn:
            org = conn.execute(
                "SELECT id, name, plan, monthly_token_quota, max_documents FROM organizations WHERE id = ?",
                (tenant_id,),
            ).fetchone()
            agg = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_operations,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0.0) AS total_cost_usd
                FROM usage_ledger
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
            doc_count_row = conn.execute(
                "SELECT COUNT(*) AS doc_count FROM documents_meta WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            by_model_rows = conn.execute(
                """
                SELECT model,
                       COUNT(*) AS calls,
                       COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens,
                       COALESCE(SUM(estimated_cost_usd), 0.0) AS cost_usd
                FROM usage_ledger
                WHERE tenant_id = ?
                GROUP BY model
                """,
                (tenant_id,),
            ).fetchall()
        total_tokens = int(agg["total_prompt_tokens"]) + int(agg["total_completion_tokens"])
        quota_tokens = int(org["monthly_token_quota"]) if org else 250_000
        return {
            "tenant_id": tenant_id,
            "org_name": org["name"] if org else tenant_id,
            "plan": org["plan"] if org else "free",
            "total_operations": int(agg["total_operations"]),
            "prompt_tokens": int(agg["total_prompt_tokens"]),
            "completion_tokens": int(agg["total_completion_tokens"]),
            "total_tokens": total_tokens,
            "monthly_token_quota": quota_tokens,
            "quota_utilization_pct": round((total_tokens / max(1, quota_tokens)) * 100.0, 2),
            "estimated_cost_usd": round(float(agg["total_cost_usd"]), 6),
            "documents_count": int(doc_count_row["doc_count"]),
            "max_documents": int(org["max_documents"]) if org else 10,
            "by_model": [dict(r) for r in by_model_rows],
        }

    def record_audit_log(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        metadata: dict | None = None,
        ip_address: str = "",
        request_id: str = "",
    ) -> None:
        log_id = f"aud_{uuid.uuid4().hex[:14]}"
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs
                (id, tenant_id, actor_id, action, resource_type, resource_id, metadata_json, ip_address, request_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    tenant_id,
                    actor_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    ip_address,
                    request_id,
                    now,
                ),
            )

    def list_audit_logs(self, tenant_id: str, limit: int = 100) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, tenant_id, actor_id, action, resource_type, resource_id,
                       metadata_json, ip_address, request_id, created_at
                FROM audit_logs
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (tenant_id, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "tenant_id": r["tenant_id"],
                "actor_id": r["actor_id"],
                "action": r["action"],
                "resource_type": r["resource_type"],
                "resource_id": r["resource_id"],
                "metadata": json.loads(r["metadata_json"] or "{}"),
                "ip_address": r["ip_address"],
                "request_id": r["request_id"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def record_feedback(
        self,
        *,
        tenant_id: str,
        user_id: str,
        trace_id: str,
        score: int,
        message_id: str = "",
        comment: str = "",
    ) -> dict:
        fb_id = f"fb_{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, tenant_id, user_id, trace_id, message_id, score, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fb_id, tenant_id, user_id, trace_id, message_id, int(score), comment, now),
            )
        return {
            "id": fb_id,
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "message_id": message_id,
            "score": int(score),
            "comment": comment,
            "created_at": now,
        }

