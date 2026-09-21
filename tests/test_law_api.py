from pathlib import Path

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


def make_settings(tmp_path: Path) -> Settings:
    """Real qdrant-local storage, isolated temp dirs — exercises the actual provider."""
    return Settings(
        upload_dir=tmp_path / "uploads",
        pending_dir=tmp_path / "pending",
        storage_provider="qdrant-local",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="law_chunks_test",
    )


def _client(tmp_path: Path, **overrides) -> TestClient:
    return TestClient(
        create_app(settings=make_settings(tmp_path), enable_tracking=False, **overrides)
    )


def test_health_endpoint(tmp_path: Path):
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf(tmp_path: Path):
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Only PDF uploads are supported"


def test_upload_only_extracts_and_chunks(tmp_path: Path):
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    embedder = FakeEmbedder()
    with _client(tmp_path, embedder=embedder) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["articles"] == 1149
        assert payload["chunks"] > 0
        assert embedder.calls == 0


def test_embed_route_runs_model_and_stores_vectors_in_qdrant(tmp_path: Path):
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    embedder = FakeEmbedder()
    with _client(tmp_path, embedder=embedder) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
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


def test_ask_endpoint_returns_grounded_answer(tmp_path: Path):
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    with _client(tmp_path, embedder=FakeEmbedder(), llm=FakeLLMProvider()) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
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


def test_document_ids_are_sequential_across_restarts(tmp_path: Path):
    """Upload with one app instance, then a fresh instance over the same
    Qdrant store must NOT reuse document id 1."""
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
        )
        assert first.status_code == 201
        assert first.json()["document_id"] == 1

    # New app instance over the same persisted Qdrant folder.
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        second = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law2.pdf", source.read_bytes(), "application/pdf")},
        )
        assert second.status_code == 201
        assert second.json()["document_id"] == 2


def test_document_ids_survive_restarts_without_embedding(tmp_path: Path):
    """Upload, restart the app WITHOUT embedding, then upload again: the
    persisted pending file must force a fresh id (never reuse id 1)."""
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
        )
        assert first.status_code == 201
        assert first.json()["document_id"] == 1

    # Restart — no embed happened; only the pending file knows about doc 1.
    with _client(tmp_path, embedder=FakeEmbedder()) as client:
        second = client.post(
            "/api/v1/documents/upload",
            files={"file": ("law2.pdf", source.read_bytes(), "application/pdf")},
        )
        assert second.status_code == 201
        assert second.json()["document_id"] == 2
