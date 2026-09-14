from pathlib import Path

from fastapi.testclient import TestClient

from law_api.main import create_app
from law_api.providers.storage.in_memory_provider import InMemoryStorageProvider


class FakeEmbedder:
    name = "test-embedder"
    model_id = "test-embedder"

    def __init__(self):
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class FakeLLMProvider:
    name = "fake-llm"
    model_id = "fake-llm-model"

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512):
        return "Fake answer citing Article 147."


def test_health_endpoint(tmp_path: Path):
    client = TestClient(create_app(embedder=FakeEmbedder(), storage=InMemoryStorageProvider()))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf(tmp_path: Path):
    client = TestClient(create_app(embedder=FakeEmbedder(), storage=InMemoryStorageProvider()))
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("law.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF uploads are supported"


def test_upload_only_extracts_and_chunks(tmp_path: Path):
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    embedder = FakeEmbedder()
    client = TestClient(create_app(embedder=embedder, storage=InMemoryStorageProvider()))
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["articles"] == 1149
    assert payload["chunks"] > 0
    assert embedder.calls == 0


def test_embed_route_runs_model_and_stores_vectors(tmp_path: Path):
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    embedder = FakeEmbedder()
    client = TestClient(create_app(embedder=embedder, storage=InMemoryStorageProvider()))
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


def test_ask_endpoint_returns_grounded_answer(tmp_path: Path):
    source = Path("src/data/raw/egyptian_civil_code.pdf")
    embedder = FakeEmbedder()
    storage = InMemoryStorageProvider()
    client = TestClient(create_app(embedder=embedder, storage=storage, llm=FakeLLMProvider()))
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("law.pdf", source.read_bytes(), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    client.post(f"/api/v1/documents/{document_id}/embed")

    search = client.post(
        f"/api/v1/documents/{document_id}/search",
        json={"question": "What does article 147 say?", "top_k": 3},
    )
    assert search.status_code == 200, search.text
    assert len(search.json()["results"]) <= 3

    ask = client.post(
        f"/api/v1/documents/{document_id}/ask",
        json={"question": "What does article 147 say?", "top_k": 3},
    )
    assert ask.status_code == 200, ask.text
    payload = ask.json()
    assert payload["answer"] == "Fake answer citing Article 147."
    assert payload["llm_model"] == "fake-llm-model"
    assert len(payload["sources"]) <= 3


def test_ask_rejects_unknown_document(tmp_path: Path):
    client = TestClient(
        create_app(embedder=FakeEmbedder(), storage=InMemoryStorageProvider(), llm=FakeLLMProvider())
    )
    response = client.post("/api/v1/documents/999/ask", json={"question": "Anything?"})
    assert response.status_code == 404