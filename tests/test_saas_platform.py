"""Comprehensive enterprise SaaS test suite: auth, multi-tenancy, jobs, conversations, metering."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from law_api.config import Settings
from law_api.main import create_app
from law_api.stores.llm.providers import MockLLMProvider


class FastEmbedder:
    name = "fast-embedder"
    model_id = "fast-embedder-v1"

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 97) + 1.0, float(t.lower().count("contract") + 1)] for t in texts]


def _make_mini_pdf(text: str = "Article 147 The contract makes the law of the parties.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 60), text)
    data = doc.tobytes()
    doc.close()
    return data


def _seed_mini_processed(tmp_path: Path, pdf_bytes: bytes) -> None:
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    target = tmp_path / "uploads" / "processed" / digest
    target.mkdir(parents=True, exist_ok=True)
    articles = [
        {
            "article_number": 147,
            "book": "Obligations",
            "chapter": "Contracts",
            "section": "Effects",
            "topic": "Binding Force",
            "text_ar": "العقد شريعة المتعاقدين.",
            "text_en": "The contract makes the law of the parties.",
            "is_repealed": False,
            "source_page": 1,
            "citation": "Egyptian Civil Code, Article 147",
        }
    ]
    chunks = [
        {
            "chunk_id": "147:paragraph:1",
            "article_number": 147,
            "book": "Obligations",
            "chapter": "Contracts",
            "section": "Effects",
            "topic": "Binding Force",
            "text_ar": "العقد شريعة المتعاقدين.",
            "text_en": "The contract makes the law of the parties.",
            "chunk_text": "The contract makes the law of the parties.",
            "chunk_number": 1,
            "is_repealed": False,
            "source_page": 1,
            "citation": "Egyptian Civil Code, Article 147",
        }
    ]
    (target / "law_articles.jsonl").write_text(
        "\n".join(json.dumps(a) for a in articles) + "\n", encoding="utf-8"
    )
    (target / "law_chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    (target / ".source_hash").write_text(digest, encoding="utf-8")


def _saas_client(tmp_path: Path, *, auth_required: bool = True, rate_limit_rpm: int = 120) -> TestClient:
    settings = Settings(
        upload_dir=tmp_path / "uploads",
        pending_dir=tmp_path / "pending",
        storage_provider="qdrant-local",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="law_saas_test",
        auth_required=auth_required,
        jwt_secret_key="test-enterprise-secret-key",
        rate_limit_rpm=rate_limit_rpm,
        hybrid_search_enabled=True,
        rag_rerank_enabled=False,
    )
    app = create_app(
        settings=settings,
        embedder=FastEmbedder(),
        llm=MockLLMProvider("mock-saas-llm"),
        enable_tracking=False,
    )
    return TestClient(app)


def test_auth_required_blocks_anonymous_and_supports_jwt_and_api_keys(tmp_path: Path):
    with _saas_client(tmp_path, auth_required=True) as client:
        # Unauthenticated request returns 401 RFC 9457 Problem Details
        unauth = client.get("/api/v1/documents")
        assert unauth.status_code == 401
        assert unauth.json()["error_code"] == "AUTHENTICATION_REQUIRED"
        assert "X-Request-ID" in unauth.headers

        # Register Org A
        reg = client.post(
            "/api/v1/auth/register",
            json={
                "org_name": "Acme Legal LLP",
                "email": "partner@acmelegal.com",
                "password": "SuperSecurePassword!1",
                "full_name": "Jane Partner",
                "plan": "enterprise",
            },
        )
        assert reg.status_code == 201, reg.text
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["org_name"] == "Acme Legal LLP"

        # Login with password
        login = client.post(
            "/api/v1/auth/login",
            json={"email": "partner@acmelegal.com", "password": "SuperSecurePassword!1"},
        )
        assert login.status_code == 200
        assert login.json()["access_token"]

        # Create scoped API key
        key_res = client.post(
            "/api/v1/tenants/api-keys",
            headers=headers,
            json={"name": "prod-sdk-key", "scopes": ["read", "write"]},
        )
        assert key_res.status_code == 201
        api_key = key_res.json()["api_key"]
        key_id = key_res.json()["id"]
        assert api_key.startswith("lrag_live_")

        # Authenticate via X-API-Key
        me_key = client.get("/api/v1/auth/me", headers={"X-API-Key": api_key})
        assert me_key.status_code == 200
        assert me_key.json()["auth_type"] == "api_key"

        # Revoke API key and confirm immediate 401
        del_res = client.delete(f"/api/v1/tenants/api-keys/{key_id}", headers=headers)
        assert del_res.status_code == 204
        revoked_res = client.get("/api/v1/auth/me", headers={"X-API-Key": api_key})
        assert revoked_res.status_code == 401


def test_strict_multi_tenant_isolation_and_conversations_and_metering(tmp_path: Path):
    pdf_bytes = _make_mini_pdf()
    _seed_mini_processed(tmp_path, pdf_bytes)
    with _saas_client(tmp_path, auth_required=True) as client:
        reg_a = client.post(
            "/api/v1/auth/register",
            json={
                "org_name": "Alpha Law",
                "email": "alpha@law.io",
                "password": "Password123!",
            },
        ).json()
        reg_b = client.post(
            "/api/v1/auth/register",
            json={
                "org_name": "Beta Law",
                "email": "beta@law.io",
                "password": "Password123!",
            },
        ).json()
        hdr_a = {"Authorization": f"Bearer {reg_a['access_token']}"}
        hdr_b = {"Authorization": f"Bearer {reg_b['access_token']}"}

        # Tenant A runs async ingestion (HTTP 202) with Idempotency-Key
        ingest = client.post(
            "/api/v1/documents/ingest",
            headers={**hdr_a, "Idempotency-Key": "idem-alpha-1"},
            files={"file": ("alpha_code.pdf", pdf_bytes, "application/pdf")},
        )
        assert ingest.status_code == 202, ingest.text
        job_id = ingest.json()["id"]

        # Duplicate Idempotency-Key returns same job id
        ingest_dup = client.post(
            "/api/v1/documents/ingest",
            headers={**hdr_a, "Idempotency-Key": "idem-alpha-1"},
            files={"file": ("alpha_code.pdf", pdf_bytes, "application/pdf")},
        )
        assert ingest_dup.status_code == 202
        assert ingest_dup.json()["id"] == job_id

        # Poll job until READY
        job_status = None
        for _ in range(40):
            job_res = client.get(f"/api/v1/documents/jobs/{job_id}", headers=hdr_a)
            assert job_res.status_code == 200
            job_status = job_res.json()
            if job_status["status"] in ("READY", "FAILED"):
                break
            time.sleep(0.05)
        assert job_status is not None
        assert job_status["status"] == "READY", job_status
        doc_a_id = job_status["document_id"]

        # Tenant B cannot see Tenant A's job, document list, search results, or delete Doc A
        assert client.get(f"/api/v1/documents/jobs/{job_id}", headers=hdr_b).status_code == 404
        assert client.get("/api/v1/documents", headers=hdr_b).json()["results"] == []
        assert client.delete(f"/api/v1/documents/{doc_a_id}", headers=hdr_b).status_code == 404
        assert (
            client.post(
                "/api/v1/search",
                headers=hdr_b,
                json={"question": "contract", "document_id": doc_a_id},
            ).status_code
            == 404
        )
        assert (
            client.post("/api/v1/search", headers=hdr_b, json={"question": "contract"}).status_code
            == 422
        )

        # Tenant A creates a multi-turn conversation and asks two turns
        conv_res = client.post(
            "/api/v1/conversations",
            headers=hdr_a,
            json={"title": "Contract Binding Force", "document_id": doc_a_id},
        )
        assert conv_res.status_code == 201
        conv_id = conv_res.json()["id"]

        turn_1 = client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            headers=hdr_a,
            json={"question": "What does Article 147 say about a contract?", "top_k": 3},
        )
        assert turn_1.status_code == 200, turn_1.text
        trace_id = turn_1.json()["assistant_message"]["trace_id"]
        assert "Article 147" in turn_1.json()["assistant_message"]["content"]

        turn_2 = client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            headers=hdr_a,
            json={"question": "Does it apply to parties?", "top_k": 3},
        )
        assert turn_2.status_code == 200

        # Tenant B cannot read Tenant A's conversation
        assert client.get(f"/api/v1/conversations/{conv_id}", headers=hdr_b).status_code == 404

        # Tenant A submits feedback
        fb = client.post(
            "/api/v1/feedback",
            headers=hdr_a,
            json={"trace_id": trace_id, "score": 1, "comment": "Accurate citation"},
        )
        assert fb.status_code == 201

        # Tenant A usage & audit logs reflect operations; Tenant B usage is zero
        usage_a = client.get("/api/v1/tenants/usage", headers=hdr_a).json()
        usage_b = client.get("/api/v1/tenants/usage", headers=hdr_b).json()
        assert usage_a["total_operations"] >= 3
        assert usage_a["total_tokens"] > 0
        assert usage_a["documents_count"] == 1
        assert usage_b["total_operations"] == 0
        assert usage_b["documents_count"] == 0

        audit_a = client.get("/api/v1/tenants/audit-logs", headers=hdr_a).json()["results"]
        actions = {row["action"] for row in audit_a}
        assert {"organization.registered", "document.ingested", "feedback.submitted"} <= actions


def test_sliding_window_rate_limit_enforced(tmp_path: Path):
    with _saas_client(tmp_path, auth_required=False, rate_limit_rpm=3) as client:
        r1 = client.get("/api/v1/documents")
        r2 = client.get("/api/v1/documents")
        r3 = client.get("/api/v1/documents")
        r4 = client.get("/api/v1/documents")
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Limit"] == "3"
        assert r2.status_code == 200
        assert r3.status_code == 200
        assert r4.status_code == 429
        assert r4.json()["error_code"] == "RATE_LIMIT_EXCEEDED"


def test_orphaned_ingestion_job_recovers_on_reupload(tmp_path: Path):
    pdf_bytes = _make_mini_pdf()
    _seed_mini_processed(tmp_path, pdf_bytes)
    with _saas_client(tmp_path, auth_required=False) as client:
        # Trigger lifespan/services initialization
        assert client.get("/api/v1/health").status_code == 200
        store = client.app.state.tenant_store
        stale_job, _ = store.create_ingestion_job(
            tenant_id="org_default",
            filename="alpha_code.pdf",
            idempotency_key="stale-key-1",
        )
        store.update_ingestion_job(
            stale_job["id"],
            status="PROCESSING",
            stage="embedding",
            progress_pct=60,
        )

        # Re-uploading with the same Idempotency-Key must reset/re-run and reach READY
        resp = client.post(
            "/api/v1/documents/ingest",
            headers={"Idempotency-Key": "stale-key-1"},
            files={"file": ("alpha_code.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["id"]

        final_job = None
        for _ in range(40):
            polled = client.get(f"/api/v1/documents/jobs/{job_id}").json()
            final_job = polled
            if polled["status"] in ("READY", "FAILED"):
                break
            time.sleep(0.05)
        assert final_job is not None
        assert final_job["status"] == "READY", final_job
        assert final_job["progress_pct"] == 100
