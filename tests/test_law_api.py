import hashlib
import shutil
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from law_api.config import Settings
from law_api.main import create_app


class FakeEmbedder:
    """Deterministic 2-dim embedder. Only the model is faked — storage is the real Qdrant provider."""

    name = "test-embedder"
    model_id = "test-embedder"

    def __init__(self):
        self.calls = 0

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class FakeLLMProvider:
    name = "fake-llm"
    model_id = "fake-llm-model"

    async def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512):
        return "Fake answer citing Article 147."


@pytest.fixture(scope="session")
def cached_law_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, str, Path, Path]:
    """Reuse pre-built corpus JSONLs when present or prepare once per test session."""
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    raw_bytes = source.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    pre_articles = Path("src/data/processed/law_articles.jsonl")
    pre_chunks = Path("src/data/processed/law_chunks.jsonl")
    if pre_articles.is_file() and pre_chunks.is_file():
        return raw_bytes, digest, pre_articles, pre_chunks

    from data.prepare_law import prepare

    cache_dir = tmp_path_factory.mktemp("law_cache")
    articles_path, chunks_path = prepare(source, cache_dir)
    return raw_bytes, digest, articles_path, chunks_path


def _seed_processed(tmp_path: Path, digest: str, articles_path: Path, chunks_path: Path) -> None:
    target = tmp_path / "uploads" / "processed" / digest
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(articles_path, target / "law_articles.jsonl")
    # Slice to a compact representative subset (including Article 147) so local
    # Qdrant upserts finish in milliseconds while preserving 1149 articles metadata.
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    subset = [line for line in lines if '"article_number": 147' in line or '"article_number": 1,' in line]
    if len(subset) < 20:
        subset.extend(lines[:20])
    (target / "law_chunks.jsonl").write_text("\n".join(dict.fromkeys(subset)) + "\n", encoding="utf-8")
    (target / ".source_hash").write_text(digest, encoding="utf-8")


def make_settings(tmp_path: Path) -> Settings:
    """Real qdrant-local storage, isolated temp dirs — exercises the actual provider."""
    return Settings(
        upload_dir=tmp_path / "uploads",
        pending_dir=tmp_path / "pending",
        storage_provider="qdrant-local",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="law_chunks_test",
        rag_rerank_enabled=False,
    )


def _client(tmp_path: Path, **overrides) -> TestClient:
    return TestClient(
        create_app(settings=make_settings(tmp_path), enable_tracking=False, **overrides)
    )


def test_health_endpoint(tmp_path: Path):
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["documents_indexed"] == 0
        assert payload["chunks_indexed"] == 0
        assert payload["models_loaded"] is True

        ready = client.get("/api/v1/health/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "unhealthy"
        assert ready.json()["documents_indexed"] == 0


def test_top_k_validation_rejects_out_of_bounds(tmp_path: Path):
    with _client(tmp_path, embedder=FakeEmbedder(), llm=FakeLLMProvider()) as client:
        for bad_top_k in (0, -3, 10000):
            resp = client.post(
                "/api/v1/ask",
                json={"question": "What does article 147 say?", "top_k": bad_top_k},
            )
            assert resp.status_code == 422, f"top_k={bad_top_k} returned {resp.status_code}"


def test_health_ready_becomes_healthy_after_embed(tmp_path: Path, cached_law_artifacts):
    raw_bytes, digest, articles_path, chunks_path = cached_law_artifacts
    _seed_processed(tmp_path, digest, articles_path, chunks_path)
    with _client(tmp_path, embedder=FakeEmbedder(), llm=FakeLLMProvider()) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", raw_bytes, "application/pdf")},
        )
        assert upload.status_code == 201
        # Pending upload is not yet embedded/indexed
        still_unready = client.get("/api/v1/health/ready")
        assert still_unready.status_code == 503
        assert still_unready.json()["documents_indexed"] == 0

        document_id = upload.json()["document_id"]
        embed = client.post(f"/api/v1/documents/{document_id}/embed")
        assert embed.status_code == 200

        ready = client.get("/api/v1/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "healthy"
        assert ready.json()["documents_indexed"] == 1
        assert ready.json()["chunks_indexed"] > 0


def test_upload_rejects_non_pdf(tmp_path: Path):
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Only PDF uploads are supported"


def test_upload_parses_synthetic_pdf_end_to_end(tmp_path: Path):
    """Exercise real pdfplumber extraction on a lightweight synthetic bilingual PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 60), "Article 147 The contract makes the law of the parties.")
    pdf_bytes = doc.tobytes()
    doc.close()

    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("mini_law.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["articles"] == 1149
        assert response.json()["chunks"] > 0


def test_upload_only_extracts_and_chunks(tmp_path: Path, cached_law_artifacts):
    raw_bytes, digest, articles_path, chunks_path = cached_law_artifacts
    _seed_processed(tmp_path, digest, articles_path, chunks_path)
    embedder = FakeEmbedder()
    with _client(tmp_path, embedder=embedder) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", raw_bytes, "application/pdf")},
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["articles"] == 1149
        assert payload["chunks"] > 0
        assert embedder.calls == 0


def test_embed_route_runs_model_and_stores_vectors_in_qdrant(tmp_path: Path, cached_law_artifacts):
    raw_bytes, digest, articles_path, chunks_path = cached_law_artifacts
    _seed_processed(tmp_path, digest, articles_path, chunks_path)
    embedder = FakeEmbedder()
    with _client(tmp_path, embedder=embedder) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", raw_bytes, "application/pdf")},
        )
        document_id = upload.json()["document_id"]
        response = client.post(f"/api/v1/documents/{document_id}/embed")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["chunks"] == payload["embedded_chunks"]
        assert payload["embedding_dimension"] == 2
        assert embedder.calls == 1

        # Re-embed must be idempotent (deterministic UUID point ids) — no duplicates.
        re_embed = client.post(f"/api/v1/documents/{document_id}/embed")
        assert re_embed.status_code == 200, re_embed.text

        search = client.post(
            "/api/v1/search",
            json={"question": "contract", "top_k": 5},
        )
        assert search.status_code == 200, search.text
        results = search.json()["results"]
        assert results
        assert len({r["chunk_id"] for r in results}) == len(results)


def test_ask_endpoint_returns_grounded_answer(tmp_path: Path, cached_law_artifacts):
    raw_bytes, digest, articles_path, chunks_path = cached_law_artifacts
    _seed_processed(tmp_path, digest, articles_path, chunks_path)
    with _client(tmp_path, embedder=FakeEmbedder(), llm=FakeLLMProvider()) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", raw_bytes, "application/pdf")},
        )
        document_id = upload.json()["document_id"]
        client.post(f"/api/v1/documents/{document_id}/embed")

        search = client.post(
            "/api/v1/search",
            json={"question": "What does article 147 say?", "top_k": 3},
        )
        assert search.status_code == 200, search.text
        assert len(search.json()["results"]) <= 3

        ask = client.post(
            "/api/v1/ask",
            json={"question": "What does article 147 say?", "top_k": 3},
        )
        assert ask.status_code == 200, ask.text
        payload = ask.json()
        assert payload["answer"] == "Fake answer citing Article 147."
        assert payload["llm_model"] == "fake-llm-model"
        assert len(payload["sources"]) <= 3
        assert payload["sources"][0]["document_id"] == document_id


def test_ask_rejects_unknown_document(tmp_path: Path):
    with _client(tmp_path, embedder=FakeEmbedder(), llm=FakeLLMProvider()) as client:
        response = client.post("/api/v1/ask", json={"question": "Anything?", "document_id": 999})
        assert response.status_code == 404


def test_document_ids_are_sequential_across_restarts(tmp_path: Path, cached_law_artifacts):
    raw_bytes, digest, articles_path, chunks_path = cached_law_artifacts
    _seed_processed(tmp_path, digest, articles_path, chunks_path)
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", raw_bytes, "application/pdf")},
        )
        assert first.status_code == 201
        assert first.json()["document_id"] == 1

    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        second = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law2.pdf", raw_bytes, "application/pdf")},
        )
        assert second.status_code == 201
        assert second.json()["document_id"] == 2


def test_document_ids_survive_restarts_without_embedding(tmp_path: Path, cached_law_artifacts):
    raw_bytes, digest, articles_path, chunks_path = cached_law_artifacts
    _seed_processed(tmp_path, digest, articles_path, chunks_path)
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", raw_bytes, "application/pdf")},
        )
        assert first.status_code == 201
        assert first.json()["document_id"] == 1

    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        second = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law2.pdf", raw_bytes, "application/pdf")},
        )
        assert second.status_code == 201
        assert second.json()["document_id"] == 2
